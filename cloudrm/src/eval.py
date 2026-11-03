import argparse, csv, os
import numpy as np
from skimage.color import rgb2gray
from skimage.feature import canny
from datasets import PairDataset, load_rgb
import metrics

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--ids", required=True)
p.add_argument("--pred_dir", required=True)
p.add_argument("--out", required=True)
p.add_argument("--cloudy_dir", default="cloud")
p.add_argument("--clear_dir", default="label")
p.add_argument("--mask_dir", default=None)
p.add_argument("--ext", default=".png")
a = p.parse_args()

ds = PairDataset(a.root, a.ids, a.cloudy_dir, a.clear_dir, a.mask_dir, a.ext)
rows = []
for i in range(len(ds)):
    s = ds[i]
    cloudy = s["cloudy"].numpy().transpose(1, 2, 0)
    clear = s["clear"].numpy().transpose(1, 2, 0)
    pred = load_rgb(os.path.join(a.pred_dir, s["id"] + ".png"))
    rows.append({
        "id": s["id"],
        "psnr": metrics.psnr(pred, clear),
        "ssim": metrics.ssim(pred, clear),
        "lpips": metrics.lpips_score(pred, clear),
        "sam": metrics.sam(pred, clear),
        "coverage": float(s["mask"].mean()),
        "brightness": float(cloudy.mean()),
        "edge_density": float(canny(rgb2gray(cloudy)).mean()),
    })
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(len(rows), "images ->", a.out)
