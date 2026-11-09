"""
Three-way comparison on EuroSAT, full 4,050-tile test split, one seed.

Classifiers (ImageNet ResNet-18, full fine-tuning, best-validation epoch):
  clean      : trained on clean tiles
  mask_aware : trained on masked tiles (shape-mask family, black fill)
  fill_aware : trained on filled tiles (nearest neighbour, Telea, mean fill, scratch U-Net; 20% clean)
Test inputs: cloudy, masked_zero, mean_fill, nearest_neighbor, telea, unet (thick clouds).

Uses the training and evaluation code from ablate_classifier_finetune.py, so the cloud generator,
masks and classifier settings match the earlier ablation.

Writes: results/threeway_seed{seed}.csv
Run:    python cloud_removal_project/src/threeway_comparison.py --seed 0
Smoke:  python cloud_removal_project/src/threeway_comparison.py --seed 0 --max_train 1000 --n_test 200
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd, torch

import ablate_classifier_finetune as A
from cloud_simulation import CloudSimulator
from inpainting_baselines import mean_fill, nearest_neighbor_fill, telea_inpaint
from unet_inpaint import SmallUNetInpainter

FILLS = ["mean_fill", "nearest_neighbor", "telea", "unet"]


def fill_all(cloudy, masks, unet, dev):
    """All fill outputs for one set of cloudy tiles, keyed by method name."""
    out = {"mean_fill": [], "nearest_neighbor": [], "telea": []}
    for c, m in zip(cloudy, masks):
        out["mean_fill"].append(mean_fill(c, m))
        out["nearest_neighbor"].append(nearest_neighbor_fill(c, m))
        out["telea"].append(telea_inpaint(c, m))
    out = {k: np.stack(v) for k, v in out.items()}
    out["unet"] = A.unet_fill(unet, cloudy, masks, dev)
    return out


def build_fill_set(X, sim, unet, dev, rng, clean_frac):
    """Filled tiles for classifier training. Each tile gets a random bin, a shape mask, and a random fill."""
    bins = list(A.BINS.values())
    masks, cloudy = [], []
    for img in X:
        b = bins[rng.integers(len(bins))]
        m = sim.generate_shape_mask(b, rng=rng)
        c, _ = sim.apply_cloud(img, m, condition="thick", rng=rng)
        masks.append(m)
        cloudy.append(c)
    cloudy, masks = np.stack(cloudy), np.stack(masks)
    fills = fill_all(cloudy, masks, unet, dev)
    choice = rng.integers(len(FILLS), size=len(X))
    out = np.empty_like(X)
    for i, k in enumerate(choice):
        out[i] = fills[FILLS[k]][i]
    keep = rng.random(len(X)) < clean_frac
    out[keep] = X[keep]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--n_test", type=int, default=None)
    ap.add_argument("--scratch", action="store_true", help="no ImageNet weights (smoke test only)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}  seed: {args.seed}")

    sp = "cloud_removal_project/data/splits"
    Xtr, ytr = A.load_split(f"{sp}/train.csv", args.max_train)
    Xva, yva = A.load_split(f"{sp}/val.csv", 2000)
    Xte, yte = A.load_split(f"{sp}/test.csv", args.n_test)
    print(f"train {len(Xtr)}  val {len(Xva)}  test {len(Xte)}")

    sim = CloudSimulator(64, 64)
    unet = SmallUNetInpainter().to(dev)
    unet.load_state_dict(torch.load("cloud_removal_project/models/unet_inpaint.pth", map_location=dev))
    unet.eval()
    prep = A.make_prep(dev)

    # Shared shape-mask family for masked training and masked validation (same as the ablation).
    mrng = np.random.default_rng(1234)
    mask_pool = {b: np.stack([sim.generate_shape_mask(r, rng=mrng) for _ in range(1500)])
                 for b, r in A.BINS.items()}
    vr = np.random.default_rng(99)
    vpool = mask_pool["30-60%"][vr.integers(len(mask_pool["30-60%"]), size=len(Xva))]
    Xva_m = (Xva * (1 - vpool[..., None])).astype(np.uint8)

    # Filled training and validation sets for the fill-aware classifier.
    rng = np.random.default_rng(args.seed)
    print("building filled training set ...")
    Xtr_f = build_fill_set(Xtr, sim, unet, dev, rng, clean_frac=0.2)
    Xva_f = build_fill_set(Xva, sim, unet, dev, np.random.default_rng(777), clean_frac=0.0)

    print("\n=== clean ===")
    clf = {"clean": A.train_one("full", "clean", Xtr, ytr, Xva, yva, Xva_m, mask_pool, args, dev, prep)}
    print("\n=== mask_aware ===")
    clf["mask_aware"] = A.train_one("full", "mask_aware", Xtr, ytr, Xva, yva, Xva_m, mask_pool, args, dev, prep)
    print("\n=== fill_aware ===")
    clf["fill_aware"] = A.train_one("full", "clean", Xtr_f, ytr, Xva_f, yva, Xva_m, mask_pool, args, dev, prep)

    print("\nbuilding test inputs ...")
    sets = A.build_eval_sets(Xte, sim, unet, dev)

    rows = []
    for cname, model in clf.items():
        acc = float(A.accuracy(model, sets["10-30%"]["clean"], yte, prep, dev))
        rows.append(dict(seed=args.seed, classifier=cname, bin="all", input="clean",
                         acc=round(acc, 4), ci95=round(1.96 * np.sqrt(acc * (1 - acc) / len(yte)), 4), n=len(yte)))
        for bname in A.BINS:
            for inp in A.INPUTS:
                if inp == "clean":
                    continue
                acc = float(A.accuracy(model, sets[bname][inp], yte, prep, dev))
                rows.append(dict(seed=args.seed, classifier=cname, bin=bname, input=inp,
                                 acc=round(acc, 4), ci95=round(1.96 * np.sqrt(acc * (1 - acc) / len(yte)), 4),
                                 n=len(yte)))
    df = pd.DataFrame(rows)
    out = Path("cloud_removal_project/results")
    out.mkdir(parents=True, exist_ok=True)
    out_csv = out / f"threeway_seed{args.seed}.csv"
    df.to_csv(out_csv, index=False)

    pd.set_option("display.width", 200)
    for b in A.BINS:
        sub = df[df.bin == b]
        print(f"\n=== bin {b}, seed {args.seed} ===")
        print(sub.pivot_table(index="classifier", columns="input", values="acc").round(3).to_string())
    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()
