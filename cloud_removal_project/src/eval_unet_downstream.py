"""
Land-cover accuracy after U-Net fill, next to the other fills (EuroSAT, thick synthetic clouds).

Uses the SAME test tiles, masks and cloud textures as ablate_classifier_finetune.py
(2000 test tiles, rng = 42 + idx*10 + bin_index), and the saved classifiers:
  resnet18_clean.pth (clean classifier) and resnet18_mask_aware.pth (mask-aware classifier).
Inputs: cloudy, masked_zero, mean_fill, nearest_neighbor, telea, unet (models/unet_inpaint.pth).
Output: results/unet_downstream.csv and a printed table (accuracy +/- 95% binomial CI).

Run:  python cloud_removal_project/src/eval_unet_downstream.py
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image

from dataset import EuroSATDataset
from classifier import load_trained_classifier, NORM_MEAN, NORM_STD
from cloud_simulation import CloudSimulator
from inpainting_baselines import masked_input, mean_fill, nearest_neighbor_fill, telea_inpaint
from unet_inpaint import SmallUNetInpainter

BINS = {"10-30%": (0.10, 0.30), "30-60%": (0.30, 0.60), "60-80%": (0.60, 0.80)}
INPUTS = ["cloudy", "masked_zero", "mean_fill", "nearest_neighbor", "telea", "unet"]


@torch.no_grad()
def unet_fill(unet, cloudy, masks, dev, bs=256):
    out = []
    for i in range(0, len(cloudy), bs):
        x = torch.from_numpy(cloudy[i:i + bs]).to(dev).permute(0, 3, 1, 2).float() / 255.0
        m = torch.from_numpy(masks[i:i + bs]).to(dev).unsqueeze(1).float()
        o = unet(x, m).permute(0, 2, 3, 1).cpu().numpy() * 255.0
        out.append(o.clip(0, 255).astype(np.uint8))
    return np.concatenate(out)


@torch.no_grad()
def accuracy(model, imgs, y, mean, std, dev, bs=256):
    correct = 0
    for i in range(0, len(imgs), bs):
        x = torch.from_numpy(imgs[i:i + bs]).to(dev).permute(0, 3, 1, 2).float() / 255.0
        correct += (model((x - mean) / std).argmax(1).cpu().numpy() == y[i:i + bs]).sum()
    return correct / len(imgs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_test", type=int, default=2000)
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ds = EuroSATDataset("cloud_removal_project/data/splits/test.csv")
    ds.df = ds.df.sample(n=min(args.n_test, len(ds.df)), random_state=0).reset_index(drop=True)
    X = np.stack([np.array(Image.open(ds.base_dir / r).convert("RGB")) for r in ds.df["rel_path"]])
    y = ds.df["label"].astype(int).values

    unet = SmallUNetInpainter().to(dev)
    unet.load_state_dict(torch.load("cloud_removal_project/models/unet_inpaint.pth", map_location=dev))
    unet.eval()
    clfs = {"clean_clf": load_trained_classifier("cloud_removal_project/models/resnet18_clean.pth", dev),
            "mask_aware_clf": load_trained_classifier("cloud_removal_project/models/resnet18_mask_aware.pth", dev)}
    mean = torch.tensor(NORM_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=dev).view(1, 3, 1, 1)
    sim = CloudSimulator(64, 64)

    rows = []
    for bi, (bname, brange) in enumerate(BINS.items()):
        d = {k: [] for k in INPUTS if k != "unet"}
        masks = []
        for idx, img in enumerate(X):
            rng = np.random.default_rng(42 + idx * 10 + bi)  # identical to the ablation script
            mask = sim.generate_cloud_mask(brange, rng=rng)
            cloudy, _ = sim.apply_cloud(img, mask, condition="thick", rng=rng)
            masks.append(mask)
            d["cloudy"].append(cloudy)
            d["masked_zero"].append(masked_input(cloudy, mask, 0))
            d["mean_fill"].append(mean_fill(cloudy, mask))
            d["nearest_neighbor"].append(nearest_neighbor_fill(cloudy, mask))
            d["telea"].append(telea_inpaint(cloudy, mask))
        sets = {k: np.stack(v) for k, v in d.items()}
        masks = np.stack(masks)
        sets["unet"] = unet_fill(unet, sets["cloudy"], masks, dev)
        for cname, clf in clfs.items():
            for inp in INPUTS:
                acc = float(accuracy(clf, sets[inp], y, mean, std, dev))
                ci = 1.96 * np.sqrt(acc * (1 - acc) / len(y))
                rows.append(dict(bin=bname, classifier=cname, input=inp, acc=round(acc, 4), ci95=round(ci, 4), n=len(y)))
        print(f"bin {bname} done")

    df = pd.DataFrame(rows)
    out = Path("cloud_removal_project/results"); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "unet_downstream.csv", index=False)
    pd.set_option("display.width", 200)
    for cname in clfs:
        for b in BINS:
            sub = df[(df.classifier == cname) & (df["bin"] == b)].set_index("input").loc[INPUTS]
            print(f"\n=== {cname}, coverage {b} ===")
            print(sub[["acc", "ci95"]].to_string())
    print(f"\nSaved to {out / 'unet_downstream.csv'}")


if __name__ == "__main__":
    main()
