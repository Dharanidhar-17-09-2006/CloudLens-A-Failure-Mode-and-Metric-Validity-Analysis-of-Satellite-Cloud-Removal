"""
Opacity sweep (EuroSAT, 30-60% coverage): where does dehazing stop working and filling take over?

Cloud opacity alpha goes from 0.2 (thin) to 1.0 (opaque). For each alpha, compare:
cloudy_as_is, mean_fill, telea_inpaint (use the mask) and dark_channel (dehazer, uses the raw cloudy
pixels only). Metrics: LPIPS (whole tile), PSNR inside the mask, and accuracy of the clean ResNet-18
land-cover classifier. 95% intervals are mean +/- 1.96 SE over tiles.

Run:  python cloud_removal_project/src/opacity_sweep.py
Quick test:  python cloud_removal_project/src/opacity_sweep.py --n_test 100
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch, lpips
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from dataset import EuroSATDataset
from classifier import load_trained_classifier, NORM_MEAN, NORM_STD
from cloud_simulation import CloudSimulator
from inpainting_baselines import cloudy_as_is, mean_fill, telea_inpaint
from dehaze_dcp import dehaze_dcp

ALPHAS = [0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.0]
METHODS = {"cloudy_as_is": lambda c, m: cloudy_as_is(c, m), "mean_fill": mean_fill,
           "telea_inpaint": telea_inpaint, "dark_channel": lambda c, m: dehaze_dcp(c)}


def apply_opacity(clean, mask, alpha, rng):
    H, W, _ = clean.shape
    base = np.clip(rng.normal(238.0, 8.0, (H, W, 1)), 210, 255).astype(np.float32).repeat(3, -1)
    m = mask[..., None].astype(np.float32)
    c = clean.astype(np.float32)
    return np.clip(c * (1 - m) + (alpha * base + (1 - alpha) * c) * m, 0, 255).astype(np.uint8)


def ci(x):
    x = np.asarray(x, float)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_test", type=int, default=500)
    ap.add_argument("--bin_lo", type=float, default=0.30)
    ap.add_argument("--bin_hi", type=float, default=0.60)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ds = EuroSATDataset("cloud_removal_project/data/splits/test.csv")
    ds.df = ds.df.sample(n=min(args.n_test, len(ds.df)), random_state=0).reset_index(drop=True)
    X = np.stack([np.array(Image.open(ds.base_dir / r).convert("RGB")) for r in ds.df["rel_path"]])
    y = ds.df["label"].astype(int).values

    clf = load_trained_classifier(device=dev)
    lp = lpips.LPIPS(net="alex").to(dev).eval()
    mean = torch.tensor(NORM_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=dev).view(1, 3, 1, 1)
    sim = CloudSimulator(64, 64)

    def to_t(a):
        return torch.from_numpy(a).to(dev).permute(0, 3, 1, 2).float() / 255.0

    rows = []
    for alpha in ALPHAS:
        clean_l, outs, masks = [], {k: [] for k in METHODS}, []
        for idx, img in enumerate(X):
            rng = np.random.default_rng(42 + idx)  # same masks and cloud texture across alphas
            mask = sim.generate_cloud_mask((args.bin_lo, args.bin_hi), rng=rng)
            cloudy = apply_opacity(img, mask, alpha, rng)
            clean_l.append(img); masks.append(mask)
            for k, f in METHODS.items():
                outs[k].append(f(cloudy, mask))
        clean_a, masks_a = np.stack(clean_l), np.stack(masks)
        for k in METHODS:
            o = np.stack(outs[k])
            ot, ct = to_t(o), to_t(clean_a)
            with torch.no_grad():
                l = np.concatenate([lp(ot[i:i + 128] * 2 - 1, ct[i:i + 128] * 2 - 1).flatten().cpu().numpy()
                                    for i in range(0, len(o), 128)])
                pred = np.concatenate([clf((ot[i:i + 128] - mean) / std).argmax(1).cpu().numpy()
                                       for i in range(0, len(o), 128)])
            mk = masks_a[..., None].astype(bool)
            mse = np.array([((o[i].astype(float) - clean_a[i].astype(float)) ** 2)[mk[i].repeat(3, -1)].mean()
                            for i in range(len(o))])
            psnr = 10 * np.log10(255.0 ** 2 / np.maximum(mse, 1e-6))
            acc = (pred == y).astype(float)
            for metric, v in [("lpips", l), ("psnr_mask", psnr), ("accuracy", acc)]:
                m_, h_ = ci(v)
                rows.append(dict(alpha=alpha, method=k, metric=metric, mean=round(m_, 4), ci95=round(h_, 4)))
        print(f"alpha {alpha} done")

    df = pd.DataFrame(rows)
    out = Path("cloud_removal_project/results"); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "opacity_sweep.csv", index=False)

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (metric, label) in zip(axs, [("lpips", "LPIPS (lower = better)"),
                                         ("psnr_mask", "PSNR in mask, dB (higher = better)"),
                                         ("accuracy", "Land-cover accuracy (higher = better)")]):
        for k in METHODS:
            s = df[(df.method == k) & (df.metric == metric)].sort_values("alpha")
            ax.errorbar(s.alpha, s["mean"], yerr=s.ci95, marker="o", capsize=2, label=k)
        ax.set_xlabel("cloud opacity"); ax.set_ylabel(label); ax.grid(alpha=0.3)
    axs[0].legend()
    fig.suptitle(f"Opacity sweep, EuroSAT, {int(args.bin_lo * 100)}-{int(args.bin_hi * 100)}% coverage, n={len(X)}")
    fig.tight_layout()
    fig.savefig(out / "opacity_sweep.png", dpi=150)

    pd.set_option("display.width", 200)
    for metric in ["lpips", "psnr_mask", "accuracy"]:
        print(f"\n=== {metric} (mean) ===")
        print(df[df.metric == metric].pivot(index="alpha", columns="method", values="mean").to_string())
    print(f"\nSaved to {out / 'opacity_sweep.csv'} and {out / 'opacity_sweep.png'}")


if __name__ == "__main__":
    main()
