"""
Evaluate classical fills, the EuroSAT-trained U-Net and the RICE-trained inpainters on the
HELD-OUT RICE test images (real cloudy tile + real mask; ground truth = paired clear tile).

Metrics: PSNR / SSIM inside the cloud mask and LPIPS vs the clear tile (no classifier needed), plus
stable-tile agreement of the fill-aware classifier (River/SeaLake tiles excluded).
RICE1 masks are estimated (thin haze included), so treat RICE1 as secondary.

  python cloud_removal_project/src/eval_rice_inpainters.py
"""
import numpy as np, pandas as pd, torch
import rice_data as D
import real_cloud_eval as R
from real_diagnostics import probs
from unet_inpaint import SmallUNetInpainter
from rice_models import BigUNetInpainter
from metrics import MetricEvaluator

# name -> (architecture, input mode). "hole": cloud pixels replaced by white before the model (self-supervised models);
# "raw": real cloudy pixels fed as-is (paired-supervised models).
RICE_MODELS = {"rice_small": ("small", "hole"), "rice_big": ("big", "hole"), "rice_big_gan": ("big", "hole"),
               "paired_small": ("small", "raw"), "paired_big": ("big", "raw")}
BASE = ["cloudy_as_is", "mean_fill", "nearest_neighbor", "telea_inpaint", "euro_unet"]


def model_fill(net, device, cloudy, mask, mode="hole"):
    inp = cloudy.copy()
    if mode == "hole":
        inp[mask == 1] = 237
    x = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).float().div(255).to(device)
    m = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(device)
    with torch.no_grad():
        o = net(x, m)
    return (o.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)


def boot_mean(df, col, n=500, seed=0):
    g = df.groupby("image_id")[col].agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    rng = np.random.default_rng(seed)
    b = [s[i].sum() / c[i].sum() for i in (rng.integers(0, len(s), len(s)) for _ in range(n))]
    return s.sum() / c.sum(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def main():
    import argparse, cv2
    ap = argparse.ArgumentParser()
    ap.add_argument("--dilate", type=int, default=0, help="grow the mask by this many pixels before filling (metrics still use the original mask)")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = D.OUT / ("eval" if args.dilate == 0 else f"eval_dil{args.dilate}"); out.mkdir(parents=True, exist_ok=True)
    models, euro_unet = R.load_models(device)
    nets = {}
    modes = {}
    for name, (arch, mode) in RICE_MODELS.items():
        p = R.ROOT / "models" / (f"{name}.pth" if name.startswith("rice_") else f"rice_{name}.pth")
        if p.exists():
            net = (BigUNetInpainter(base=48) if arch == "big" else SmallUNetInpainter()).to(device)
            net.load_state_dict(torch.load(p, map_location=device)); nets[name] = net.eval(); modes[name] = mode
    print("RICE-trained models found:", list(nets))
    evaluator = MetricEvaluator(use_lpips=True, device=device)
    rice1, rice2, splits = D.get_data()
    methods = BASE + list(nets)

    mrows, arows, shown = [], [], {}
    for dname, items, keys in (("RICE2", rice2, splits["RICE2"]["test"]), ("RICE1", rice1, splits["RICE1"]["test"])):
        tiles = D.eval_tiles(items, keys)
        print(f"{dname}: {len(tiles)} held-out tiles from {len(keys)} images")
        outs = {m: [] for m in methods}
        for t in tiles:
            fm = t["mask"] if args.dilate == 0 else cv2.dilate(t["mask"], np.ones((2 * args.dilate + 1,) * 2, np.uint8))
            o = R.fill_all(euro_unet, device, t["cloudy"], fm)
            o["euro_unet"] = o["unet"]
            for m in BASE:
                outs[m].append(o[m])
            for name, net in nets.items():
                outs[name].append(model_fill(net, device, t["cloudy"], fm, modes[name]))
        for i, t in enumerate(tiles):
            for m in methods:
                met = evaluator.evaluate_all(t["clear"], outs[m][i], t["mask"])
                mrows.append(dict(dataset=dname, image_id=t["image_id"], bin=t["bin"], method=m,
                                  psnr_mask=met["psnr_mask"], ssim_mask=met["ssim_mask"], lpips=float(met["lpips"])))
        # fill-aware classifier: stable tiles, River/SeaLake excluded
        fa = models["fill_aware"]
        clears = [t["clear"] for t in tiles]
        rng = np.random.default_rng(0)
        deg = [np.clip(c.astype(np.float32) * rng.uniform(0.9, 1.1) + rng.normal(0, 5, c.shape), 0, 255).astype(np.uint8) for c in clears]
        pc = probs(fa, clears, device).argmax(1)
        keep = (pc == probs(fa, deg, device).argmax(1)) & ~np.isin(pc, [8, 9])
        for m in methods:
            pm = probs(fa, outs[m], device).argmax(1)
            for i, t in enumerate(tiles):
                if keep[i]:
                    arows.append(dict(dataset=dname, image_id=t["image_id"], bin=t["bin"], method=m, agree=int(pm[i] == pc[i])))
        shown[dname] = (tiles, outs)

    met = pd.DataFrame(mrows); met.to_csv(out / "fidelity_per_tile.csv", index=False)
    pd.set_option("display.width", 220)
    for col in ["lpips", "psnr_mask", "ssim_mask"]:
        print(f"\n=== {col} by coverage bin (held-out RICE tiles; LPIPS lower = better) ===")
        print(met.pivot_table(index="method", columns=["dataset", "bin"], values=col).round(3).reindex(methods).to_string())
    rows = []
    for (d, m), g in met.groupby(["dataset", "method"]):
        r = {"dataset": d, "method": m, "n_tiles": len(g)}
        for col in ["lpips", "psnr_mask", "ssim_mask"]:
            mean, lo, hi = boot_mean(g, col)
            r[col] = f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"
        rows.append(r)
    summ = pd.DataFrame(rows); summ.to_csv(out / "fidelity_summary_ci.csv", index=False)
    print("\n=== overall, 95% CI (bootstrap over images) ===\n", summ.to_string(index=False))

    ag = pd.DataFrame(arows)
    if len(ag):
        rows = []
        for (d, m), g in ag.groupby(["dataset", "method"]):
            mean, lo, hi = boot_mean(g, "agree")
            rows.append(dict(dataset=d, method=m, n_tiles=len(g), agreement_pct=f"{100 * mean:.1f} [{100 * lo:.1f}, {100 * hi:.1f}]"))
        a2 = pd.DataFrame(rows); a2.to_csv(out / "agreement_fill_aware.csv", index=False)
        print("\n=== fill-aware classifier: agreement with clear-tile prediction (stable, non-water tiles) ===\n", a2.to_string(index=False))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tiles, outs = shown["RICE2"]
    sel = np.random.default_rng(1).choice(len(tiles), min(8, len(tiles)), replace=False)
    cols = ["clear", "cloudy"] + methods
    fig, axes = plt.subplots(len(sel), len(cols), figsize=(1.9 * len(cols), 2.0 * len(sel)))
    for r, i in enumerate(sel):
        for c, name in enumerate(cols):
            img = tiles[i]["clear"] if name == "clear" else tiles[i]["cloudy"] if name == "cloudy" else outs[name][i]
            axes[r, c].imshow(img); axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(name, fontsize=7)
    plt.tight_layout(); plt.savefig(out / "rice_inpaint_examples.png", dpi=200); plt.close()
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
