"""
Tile-level bootstrap summary for metric validity (no retraining).

Reads results/metric_validity_tiles_seed{seed}.csv written by metric_validity_full.py.
Resamples TEST TILES (not rows). Each tile appears 12 times (3 bins x 4 fills), so row-level
resampling treats repeated measurements of the same tile as independent and makes the intervals too narrow.

Run: python cloud_removal_project/src/summarize_metric_validity_tiles.py --seed 0
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
df["tile"] = df.groupby(["bin", "method"]).cumcount()          # row position within each block = tile index
df["cond"] = df["bin"] + "|" + df["method"]
conds = sorted(df.cond.unique())
n_tiles = df.tile.nunique()
bins = sorted(df.bin.unique())


def matrix(col):
    return df.pivot(index="tile", columns="cond", values=col)[conds].values


H = matrix("harmful").astype(float)
cond_bin = np.array([c.split("|")[0] for c in conds])

rng = np.random.default_rng(args.seed)
rows = []
for metric in ["psnr_mask", "ssim_mask", "ssim_full", "lpips"]:
    M = matrix(metric)
    scopes = [("pooled", np.arange(len(conds)))] + [(b, np.where(cond_bin == b)[0]) for b in bins]
    for scope, cols in scopes:
        x0 = M[:, cols].ravel()
        y0 = H[:, cols].ravel()
        r0 = spearmanr(x0, y0).correlation
        boot = []
        for _ in range(args.reps):
            idx = rng.integers(0, n_tiles, n_tiles)
            boot.append(spearmanr(M[idx][:, cols].ravel(), H[idx][:, cols].ravel()).correlation)
        lo, hi = np.nanpercentile(boot, [2.5, 97.5])
        rows.append(dict(metric=metric, scope=scope, rho=round(r0, 3),
                         ci_lo=round(lo, 3), ci_hi=round(hi, 3), n_tiles=n_tiles))

out = pd.DataFrame(rows)
out_path = Path("cloud_removal_project/results") / f"metric_validity_tile_summary_seed{args.seed}.csv"
out.to_csv(out_path, index=False)
pd.set_option("display.width", 200)
print("=== tile-level bootstrap, 95% CI ===")
print(out.to_string(index=False))
print(f"\nSaved to {out_path}")
