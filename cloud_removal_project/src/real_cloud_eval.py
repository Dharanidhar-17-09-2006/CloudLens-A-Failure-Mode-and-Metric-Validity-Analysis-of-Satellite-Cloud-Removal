"""
Real-cloud check on RICE (paired cloudy / clear images).

RICE has no land-cover labels, so the metric is PREDICTION AGREEMENT:
does a classifier give the SAME prediction on the filled tile as on the paired clear tile?
The same metric is computed on synthetic clouds (EuroSAT) so real vs synthetic is apples-to-apples.

Tiles: each RICE image is cut into TILE x TILE patches, resized to 64x64 (EuroSAT input size).
Mask: RICE2 mask if --mask_dir is given, otherwise estimated from |cloudy - clear| (oracle mask,
state this in the report). Tiles with 10-80% cloud coverage are kept.

Run from the folder that contains cloud_removal_project/ :
  python cloud_removal_project/src/real_cloud_eval.py --cloud_dir <cloudy> --clear_dir <clear> [--mask_dir <mask>]
"""
import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd, cv2

ROOT = Path("cloud_removal_project")
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "results" / "real_clouds"
METHODS = ["cloudy_as_is", "masked", "mean_fill", "nearest_neighbor", "telea_inpaint", "unet"]
CLFS = ["clean", "mask_aware", "fill_aware"]
BINS = [("10-30%", 0.10, 0.30), ("30-60%", 0.30, 0.60), ("60-80%", 0.60, 0.80)]
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


# ---------------- data handling (no torch needed) ----------------
def list_images(d):
    return {p.stem: p for p in sorted(Path(d).rglob("*")) if p.suffix.lower() in IMG_EXT}

def read_rgb(p):
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read {p}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def estimate_mask(cloudy, clear, thr=25):
    d = np.abs(cloudy.astype(np.int16) - clear.astype(np.int16)).mean(axis=2)
    m = (d > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return m, d

def bin_of(cov):
    for name, lo, hi in BINS:
        if lo <= cov < hi or (name == "60-80%" and cov == hi):
            return name
    return None

def make_tiles(cloudy, clear, mask, diff, tile, size=64):
    H, W = mask.shape
    out = []
    for y in range(0, H - tile + 1, tile):
        for x in range(0, W - tile + 1, tile):
            m = mask[y:y + tile, x:x + tile]
            mt = (cv2.resize(m.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8)
            cov = float(mt.mean())
            b = bin_of(cov)
            if b is None:
                continue
            d = diff[y:y + tile, x:x + tile]
            out.append(dict(
                cloudy=cv2.resize(cloudy[y:y + tile, x:x + tile], (size, size), interpolation=cv2.INTER_AREA),
                clear=cv2.resize(clear[y:y + tile, x:x + tile], (size, size), interpolation=cv2.INTER_AREA),
                mask=mt, coverage=cov, bin=b,
                thickness=float(d[m == 1].mean()) if m.sum() else 0.0))
    return out

def real_items(cloud_dir, clear_dir, mask_dir, tile, thr, max_tiles, seed=0):
    clouds, clears = list_images(cloud_dir), list_images(clear_dir)
    masks = list_images(mask_dir) if mask_dir else {}
    keys = sorted(set(clouds) & set(clears))
    if not keys:
        raise SystemExit(f"No filename-matched pairs.\n cloud sample: {list(clouds)[:5]}\n clear sample: {list(clears)[:5]}")
    print(f"{len(keys)} cloudy/clear pairs found" + (f", {len(masks)} masks" if masks else " (estimating masks from the difference)"))
    items = []
    for k in keys:
        cl, cr = read_rgb(clouds[k]), read_rgb(clears[k])
        if cr.shape != cl.shape:
            cr = cv2.resize(cr, (cl.shape[1], cl.shape[0]), interpolation=cv2.INTER_AREA)
        diff = np.abs(cl.astype(np.int16) - cr.astype(np.int16)).mean(axis=2)
        if k in masks:
            m = cv2.imread(str(masks[k]), cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (cl.shape[1], cl.shape[0]), interpolation=cv2.INTER_NEAREST)
            m = (m > (127 if m.max() > 1 else 0)).astype(np.uint8)   # masks stored as 0/255 or 0/1
        else:
            m, _ = estimate_mask(cl, cr, thr)
        for t in make_tiles(cl, cr, m, diff, tile):
            t["image_id"] = k
            items.append(t)
    rng = np.random.default_rng(seed)
    if len(items) > max_tiles:
        items = [items[i] for i in rng.choice(len(items), max_tiles, replace=False)]
    print(f"{len(items)} real tiles kept (10-80% coverage)")
    return items

def synthetic_items(n_images, seed=0):
    from cloud_simulation import CloudSimulator
    sim = CloudSimulator(64, 64)
    df = pd.read_csv(ROOT / "data/splits/test.csv")
    df = df.sample(min(n_images, len(df)), random_state=seed).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    items = []
    for i, row in df.iterrows():
        clean = np.array(cv2.cvtColor(cv2.imread(str(ROOT / row["rel_path"])), cv2.COLOR_BGR2RGB))
        for name, lo, hi in BINS:
            m = sim.generate_cloud_mask((lo, hi), rng=rng)
            cl, _ = sim.apply_cloud(clean, m, condition="thick", rng=rng)
            d = np.abs(cl.astype(np.int16) - clean.astype(np.int16)).mean(axis=2)
            items.append(dict(cloudy=cl, clear=clean, mask=m, coverage=float(m.mean()), bin=name,
                              thickness=float(d[m == 1].mean()) if m.sum() else 0.0, image_id=f"s{i}"))
    return items

def cluster_ci(df, n_boot=500, seed=0):
    """Mean agreement with 95% CI, bootstrapping over source images (tiles of one image are correlated)."""
    g = df.groupby("image_id")["agree"].agg(["sum", "count"])
    s, c = g["sum"].values, g["count"].values
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(s), len(s))
        boots.append(s[idx].sum() / max(c[idx].sum(), 1))
    return s.sum() / c.sum(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


# ---------------- models (torch) ----------------
def load_models(device):
    import torch
    from classifier import get_classifier
    from unet_inpaint import SmallUNetInpainter
    def clf(path):
        m = get_classifier(num_classes=10, pretrained=False)
        m.load_state_dict(torch.load(ROOT / path, map_location=device))
        return m.to(device).eval()
    models = {"clean": clf("models/resnet18_clean.pth"),
              "mask_aware": clf("models/resnet18_mask_aware.pth"),
              "fill_aware": clf("models/resnet18_fill_aware.pth")}
    unet = SmallUNetInpainter().to(device)
    unet.load_state_dict(torch.load(ROOT / "models/unet_inpaint.pth", map_location=device))
    return models, unet.eval()

def predict_batch(model, imgs, device, bs=256):
    import torch
    from classifier import NORM_MEAN, NORM_STD
    mean = torch.tensor(NORM_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            x = torch.from_numpy(np.stack(imgs[i:i + bs])).permute(0, 3, 1, 2).float().div(255)
            out.append(model(((x - mean) / std).to(device)).argmax(1).cpu().numpy())
    return np.concatenate(out)

def fill_all(unet, device, cloudy, mask):
    import torch
    import inpainting_baselines as B
    with torch.no_grad():
        inp = cloudy.copy()
        inp[mask == 1] = 237            # 0.93*255: same constant the U-Net saw inside the hole during training
        x = torch.from_numpy(inp).permute(2, 0, 1).unsqueeze(0).float().div(255).to(device)
        m = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(device)
        u = (unet(x, m).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return {"cloudy_as_is": B.cloudy_as_is(cloudy, mask), "masked": B.masked_input(cloudy, mask, fill_value=0),
            "mean_fill": B.mean_fill(cloudy, mask), "nearest_neighbor": B.nearest_neighbor_fill(cloudy, mask),
            "telea_inpaint": B.telea_inpaint(cloudy, mask), "unet": u}

def agreement_rows(items, models, unet, device, source):
    outs = {m: [] for m in METHODS}
    for it in items:
        o = fill_all(unet, device, it["cloudy"], it["mask"])
        for m in METHODS:
            outs[m].append(o[m])
    rows = []
    for cn, model in models.items():
        ref = predict_batch(model, [it["clear"] for it in items], device)
        for m in METHODS:
            pred = predict_batch(model, outs[m], device)
            for it, r, p in zip(items, ref, pred):
                rows.append((source, it["image_id"], it["bin"], round(it["thickness"], 1), cn, m, int(r == p)))
    return pd.DataFrame(rows, columns=["source", "image_id", "bin", "thickness", "classifier", "method", "agree"]), outs


# ---------------- reporting ----------------
def summarize(df):
    rows = []
    for (s, c, b, m), g in df.groupby(["source", "classifier", "bin", "method"]):
        mean, lo, hi = cluster_ci(g)
        rows.append(dict(source=s, classifier=c, bin=b, method=m, n_tiles=len(g),
                         agreement_pct=100 * mean, ci_low=100 * lo, ci_high=100 * hi))
    return pd.DataFrame(rows)

def make_figures(summ, items, outs, models, device):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = [b[0] for b in BINS]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), sharey=True)
    colors = dict(zip(METHODS, plt.cm.tab10.colors))
    for ax, cn in zip(axes, CLFS):
        for m in METHODS:
            for src, ls in (("real", "-"), ("synthetic", "--")):
                s = summ[(summ.classifier == cn) & (summ.method == m) & (summ.source == src)].set_index("bin")
                xs = [b for b in bins if b in s.index]
                ax.plot(xs, [s.loc[b, "agreement_pct"] for b in xs], ls, marker="o", color=colors[m],
                        label=f"{m} ({src})" if cn == "clean" else None)
        ax.set_title(f"classifier = {cn}"); ax.set_xlabel("cloud coverage bin"); ax.set_ylim(0, 100)
    axes[0].set_ylabel("agreement with clear-tile prediction (%)")
    axes[0].legend(fontsize=6, ncol=2)
    plt.tight_layout(); plt.savefig(OUT / "real_vs_synthetic_agreement.png", dpi=250); plt.close()

    rng = np.random.default_rng(1)
    sel = rng.choice(len(items), min(8, len(items)), replace=False)
    show = ["clear", "cloudy", "mean_fill", "nearest_neighbor", "telea_inpaint", "unet"]
    fa = models["fill_aware"]
    fig, axes = plt.subplots(len(sel), len(show), figsize=(2.2 * len(show), 2.3 * len(sel)))
    for r, i in enumerate(sel):
        ref = predict_batch(fa, [items[i]["clear"]], device)[0]
        for c, name in enumerate(show):
            img = items[i]["clear"] if name == "clear" else items[i]["cloudy"] if name == "cloudy" else outs[name][i]
            ax = axes[r, c]; ax.imshow(img); ax.axis("off")
            if name in METHODS[2:]:
                ok = predict_batch(fa, [img], device)[0] == ref
                ax.set_title(f"{name}\n{'same' if ok else 'CHANGED'} pred", fontsize=7, color="green" if ok else "red")
            else:
                ax.set_title(name, fontsize=7)
    plt.tight_layout(); plt.savefig(OUT / "real_cloud_examples.png", dpi=200); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud_dir", required=True); ap.add_argument("--clear_dir", required=True)
    ap.add_argument("--mask_dir", default=None)
    ap.add_argument("--tile", type=int, default=128, help="patch size in RICE pixels before resizing to 64x64")
    ap.add_argument("--thr", type=float, default=25, help="difference threshold for estimated masks")
    ap.add_argument("--max_tiles", type=int, default=3000)
    ap.add_argument("--n_synth", type=int, default=300)
    ap.add_argument("--tag", default="rice", help="output subfolder name under results/real_clouds/")
    a = ap.parse_args()
    import torch
    global OUT
    OUT = ROOT / "results" / "real_clouds" / a.tag
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    models, unet = load_models(device)

    real = real_items(a.cloud_dir, a.clear_dir, a.mask_dir, a.tile, a.thr, a.max_tiles)
    if not real:
        raise SystemExit("No tiles with 10-80% cloud coverage. Lower --thr or check the mask / folder choice.")
    df_r, outs_r = agreement_rows(real, models, unet, device, "real")
    syn = synthetic_items(a.n_synth)
    df_s, _ = agreement_rows(syn, models, unet, device, "synthetic")

    df = pd.concat([df_r, df_s]); df.to_csv(OUT / "agreement_per_tile.csv", index=False)
    summ = summarize(df); summ.to_csv(OUT / "real_vs_synthetic_agreement.csv", index=False)
    med = df_r["thickness"].median()
    thick = df_r.assign(group=np.where(df_r["thickness"] >= med, "dense", "thin"))
    t_rows = []
    for (c, m, g), gg in thick.groupby(["classifier", "method", "group"]):
        mean, lo, hi = cluster_ci(gg)
        t_rows.append(dict(classifier=c, method=m, thickness_group=g, n=len(gg), agreement_pct=100 * mean, ci_low=100 * lo, ci_high=100 * hi))
    pd.DataFrame(t_rows).to_csv(OUT / "real_by_thickness.csv", index=False)

    make_figures(summ, real, outs_r, models, device)
    pd.set_option("display.width", 200)
    for cn in CLFS:
        print(f"\n=== agreement % with clear-tile prediction | classifier = {cn} ===")
        print(summ[summ.classifier == cn].pivot_table(index="method", columns=["source", "bin"], values="agreement_pct").round(1).to_string())
    print(f"\nSaved to {OUT}")

if __name__ == "__main__":
    main()
