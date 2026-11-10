"""
Metric validity on the full EuroSAT test split (4,050 tiles), fill-aware classifier.

For each test tile, each coverage bin (thick clouds) and each fill (mean, NN, Telea, U-Net):
  psnr_mask, ssim_mask (SSIM averaged inside the mask), ssim_full, lpips (whole tile).
  harmful flip: classifier correct on the clear tile, wrong on the filled tile.
  helpful flip: classifier wrong on the clear tile, correct on the filled tile.

Reports Spearman rho between each metric and the harmful flip (0/1), pooled and within each bin,
with bootstrap 95% CIs over tiles.  Expected sign if the metric is valid:
  psnr, ssim (higher = better)  -> negative rho with harmful flips
  lpips (lower = better)        -> positive rho with harmful flips

The fill-aware classifier is retrained here with the same recipe as threeway_comparison.py
(the three-way run did not save its models), so its numbers may differ slightly from that run.

Writes: results/metric_validity_tiles_seed{seed}.csv   (one row per tile x bin x fill)
        results/metric_validity_summary_seed{seed}.csv
Run:    python cloud_removal_project/src/metric_validity_full.py --seed 0
"""
import argparse
from pathlib import Path

import lpips
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity

import ablate_classifier_finetune as A
import threeway_comparison as T
from cloud_simulation import CloudSimulator
from unet_inpaint import SmallUNetInpainter


def predict(model, imgs, prep, dev, bs=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            x = torch.from_numpy(imgs[i:i + bs]).to(dev).permute(0, 3, 1, 2).float() / 255.0
            out.append(model(prep(x)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def lpips_scores(lp, clear, filled, dev, bs=256):
    out = []
    with torch.no_grad():
        for i in range(0, len(clear), bs):
            a = torch.from_numpy(clear[i:i + bs]).to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1
            b = torch.from_numpy(filled[i:i + bs]).to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1
            out.append(lp(a, b).flatten().cpu().numpy())
    return np.concatenate(out)


def psnr_in_mask(clear, filled, masks):
    out = np.full(len(clear), np.nan)
    for i, (a, b, m) in enumerate(zip(clear, filled, masks)):
        mm = m.astype(bool)
        if mm.any():
            mse = ((a[mm].astype(np.float64) - b[mm].astype(np.float64)) ** 2).mean()
            out[i] = 10 * np.log10(255.0 ** 2 / max(mse, 1e-6))
    return out


def ssim_scores(clear, filled, masks):
    full = np.full(len(clear), np.nan)
    inmask = np.full(len(clear), np.nan)
    for i, (a, b, m) in enumerate(zip(clear, filled, masks)):
        s, S = structural_similarity(a, b, channel_axis=2, data_range=255, full=True)
        full[i] = s
        if S.ndim == 3:
            S = S.mean(-1)
        mm = m.astype(bool)
        if mm.any():
            inmask[i] = S[mm].mean()
    return full, inmask


def rho_with_ci(x, y, reps=500, seed=0):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    r0 = spearmanr(x, y).correlation
    rng = np.random.default_rng(seed)
    n = len(x)
    boot = [spearmanr(x[i], y[i]).correlation for i in (rng.integers(0, n, n) for _ in range(reps))]
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return r0, lo, hi, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--scratch", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}  seed: {args.seed}")

    sp = "cloud_removal_project/data/splits"
    Xtr, ytr = A.load_split(f"{sp}/train.csv", None)
    Xva, yva = A.load_split(f"{sp}/val.csv", 2000)
    Xte, yte = A.load_split(f"{sp}/test.csv", None)
    print(f"train {len(Xtr)}  val {len(Xva)}  test {len(Xte)}")

    sim = CloudSimulator(64, 64)
    prep = A.make_prep(dev)
    unet = SmallUNetInpainter().to(dev)
    unet.load_state_dict(torch.load("cloud_removal_project/models/unet_inpaint.pth", map_location=dev))
    unet.eval()

    print("building filled training set and training fill-aware classifier ...")
    rng = np.random.default_rng(args.seed)
    Xtr_f = T.build_fill_set(Xtr, sim, unet, dev, rng, clean_frac=0.2)
    Xva_f = T.build_fill_set(Xva, sim, unet, dev, np.random.default_rng(777), clean_frac=0.0)
    clf = A.train_one("full", "clean", Xtr_f, ytr, Xva_f, yva, Xva, None, args, dev, prep)

    pc = predict(clf, Xte, prep, dev)
    print(f"clean-tile accuracy (test): {(pc == yte).mean():.4f}")

    lp = lpips.LPIPS(net="alex").to(dev).eval()
    frames = []
    for bi, (bname, brange) in enumerate(A.BINS.items()):
        masks, cloudy = [], []
        for idx, img in enumerate(Xte):
            r = np.random.default_rng(42 + idx * 10 + bi)   # same cloud seeds as the ablation
            m = sim.generate_cloud_mask(brange, rng=r)
            c, _ = sim.apply_cloud(img, m, condition="thick", rng=r)
            masks.append(m)
            cloudy.append(c)
        masks, cloudy = np.stack(masks), np.stack(cloudy)
        fills = T.fill_all(cloudy, masks, unet, dev)

        for meth in T.FILLS:
            F = fills[meth]
            pf = predict(clf, F, prep, dev)
            psnr = psnr_in_mask(Xte, F, masks)
            ssim_f, ssim_m = ssim_scores(Xte, F, masks)
            lp_s = lpips_scores(lp, Xte, F, dev)
            frames.append(pd.DataFrame({
                "bin": bname, "method": meth,
                "psnr_mask": psnr, "ssim_mask": ssim_m, "ssim_full": ssim_f, "lpips": lp_s,
                "harmful": ((pc == yte) & (pf != yte)).astype(int),
                "helpful": ((pc != yte) & (pf == yte)).astype(int),
                "coverage": masks.reshape(len(masks), -1).mean(1),
            }))
        print(f"bin {bname} done")

    df = pd.concat(frames, ignore_index=True)
    out = Path("cloud_removal_project/results")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / f"metric_validity_tiles_seed{args.seed}.csv", index=False)

    rows = []
    for metric in ["psnr_mask", "ssim_mask", "ssim_full", "lpips"]:
        scopes = [("pooled", df)] + [(b, df[df.bin == b]) for b in A.BINS]
        for scope, sub in scopes:
            r, lo, hi, n = rho_with_ci(sub[metric].values, sub["harmful"].values.astype(float), seed=args.seed)
            rows.append(dict(metric=metric, scope=scope, rho_vs_harmful=round(r, 3),
                             ci_lo=round(lo, 3), ci_hi=round(hi, 3), n=n))
    summ = pd.DataFrame(rows)
    summ.to_csv(out / f"metric_validity_summary_seed{args.seed}.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== Spearman rho between metric and harmful flip (95% bootstrap CI) ===")
    print(summ.to_string(index=False))

    print("\n=== harmful flip rate (%) by fill and bin ===")
    rate = df.groupby(["method", "bin"]).harmful.mean().mul(100).round(1).unstack()
    print(rate.to_string())
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
