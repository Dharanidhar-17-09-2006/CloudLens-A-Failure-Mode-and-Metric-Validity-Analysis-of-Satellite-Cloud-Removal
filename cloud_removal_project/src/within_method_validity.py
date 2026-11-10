"""
Within-method metric validity: does each metric track harmful flips inside ONE fill method?

Pooling all fills can make a metric look valid just because some fills score better on both
the metric and the flips. This checks each fill separately, using the tile-level CSV from
metric_validity_full.py (no retraining). Tile-level bootstrap 95% CIs over the 4,050 tiles.

Run: python cloud_removal_project/src/within_method_validity.py --seed 0
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ap = argparse.ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--reps", type=int, default=1000)
args = ap.parse_args()

path = Path("cloud_removal_project/results") / f"metric_validity_tiles_seed{args.seed}.csv"
df = pd.read_csv(path)
df["tile"] = df.groupby(["bin", "method"]).cumcount()
rng = np.random.default_rng(args.seed)

rows = []
for method in sorted(df.method.unique()):
    sub = df[df.method == method]
    H = sub.pivot(index="tile", columns="bin", values="harmful").values.astype(float)
    n = H.shape[0]
    for metric in ["psnr_mask", "ssim_mask", "ssim_full", "lpips"]:
        M = sub.pivot(index="tile", columns="bin", values=metric).values
        r0 = spearmanr(M.ravel(), H.ravel()).correlation
        boot = []
        for _ in range(args.reps):
            idx = rng.integers(0, n, n)
            boot.append(spearmanr(M[idx].ravel(), H[idx].ravel()).correlation)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        rows.append(dict(method=method, metric=metric, rho=round(r0, 3),
                         ci_lo=round(lo, 3), ci_hi=round(hi, 3), n_tiles=n))

out = pd.DataFrame(rows)
out_path = Path("cloud_removal_project/results") / f"within_method_summary_seed{args.seed}.csv"
out.to_csv(out_path, index=False)
pd.set_option("display.width", 200)
print("=== within-method rho (all three bins pooled within each fill), tile-level 95% CI ===")
print(out.to_string(index=False))
print(f"\nSaved to {out_path}")
