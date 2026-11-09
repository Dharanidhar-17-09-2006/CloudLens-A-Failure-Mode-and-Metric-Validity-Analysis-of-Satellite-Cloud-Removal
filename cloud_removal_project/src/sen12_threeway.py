"""
Real-cloud labelled comparison: SEN12MS-CR tiles with SEN12MS IGBP labels, one seed.

Classifiers (ImageNet ResNet-18, full fine-tuning, best-validation epoch), trained on the TRAIN split only:
  clean      : cloud-free tiles (target)
  mask_aware : cloud-free tiles with black holes from the estimated cloud masks
  fill_aware : cloudy tiles filled by NN, Telea, mean fill, scratch U-Net; 20% cloud-free
Test inputs (TEST split, real clouds): cloud_free, cloudy, masked_zero, mean_fill,
  nearest_neighbor, telea, unet. Masks are the estimated masks from build_sen12mscr_subset.py.

Labels: classes with fewer than --min_test test tiles (coverage >= 10%) are dropped from every split.
Writes: results/sen12_threeway_seed{seed}.csv
Run:    python cloud_removal_project/src/sen12_threeway.py --seed 0 --tiles cloud_removal_project/data/sen12mscr_tiles_v2.npz
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import ablate_classifier_finetune as A
import threeway_comparison as T
from unet_inpaint import SmallUNetInpainter

REAL_BINS = {"10-30%": (0.1, 0.3), "30-60%": (0.3, 0.6), "60-80%": (0.6, 0.8), "80-100%": (0.8, 1.01)}
INPUTS = ["cloud_free", "cloudy", "masked_zero", "mean_fill", "nearest_neighbor", "telea", "unet"]
CLFS = ["clean", "mask_aware", "fill_aware"]


def load_tiles(path, min_test):
    with np.load(path) as d:
        cloudy, target, mask = d["cloudy"], d["target"], d["mask"]
        label, split, cov = d["label"].astype(int), d["split"], d["coverage"]
    test_ok = (split == "test") & (cov >= 0.1)
    counts = pd.Series(label[test_ok]).value_counts()
    keep_labels = sorted(counts[counts >= min_test].index.tolist())
    print("kept classes:", keep_labels)
    print("dropped classes:", sorted(set(np.unique(label).tolist()) - set(keep_labels)))
    keep = np.isin(label, keep_labels)
    out = {}
    for s in ["train", "val", "test"]:
        m = keep & (split == s)
        out[s] = dict(cloudy=cloudy[m], target=target[m], mask=mask[m], y=label[m], cov=cov[m])
    return out


def make_mask_pool(train):
    """Masks grouped by the ablation's bin names, so A.train_one can sample from them."""
    pool = {}
    for b, (lo, hi) in A.BINS.items():
        m = train["mask"][(train["cov"] >= lo) & (train["cov"] < hi)]
        pool[b] = m if len(m) else train["mask"]
    return pool


def build_fill_train(cloudy, mask, target, unet, dev, rng, clean_frac):
    fills = T.fill_all(cloudy, mask, unet, dev)
    choice = rng.integers(len(T.FILLS), size=len(cloudy))
    out = np.empty_like(cloudy)
    for i, k in enumerate(choice):
        out[i] = fills[T.FILLS[k]][i]
    keep = rng.random(len(cloudy)) < clean_frac
    out[keep] = target[keep]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tiles", default="cloud_removal_project/data/sen12mscr_tiles_v2.npz")
    ap.add_argument("--min_test", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--scratch", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}  seed: {args.seed}")

    D = load_tiles(args.tiles, args.min_test)
    tr, va, te = D["train"], D["val"], D["test"]
    print(f"train {len(tr['y'])}  val {len(va['y'])}  test {len(te['y'])}")
    if len(tr["y"]) == 0 or len(te["y"]) == 0:
        raise SystemExit("Empty train or test split after class filtering. Lower --min_test or collect more scenes.")

    prep = A.make_prep(dev)
    mask_pool = make_mask_pool(tr)
    Xva_m = (va["target"] * (1 - va["mask"][..., None])).astype(np.uint8)

    unet = SmallUNetInpainter().to(dev)
    unet.load_state_dict(torch.load("cloud_removal_project/models/unet_inpaint.pth", map_location=dev))
    unet.eval()

    rng = np.random.default_rng(args.seed)
    print("building filled training set ...")
    Xtr_f = build_fill_train(tr["cloudy"], tr["mask"], tr["target"], unet, dev, rng, 0.2)
    Xva_f = build_fill_train(va["cloudy"], va["mask"], va["target"], unet, dev,
                             np.random.default_rng(777), 0.0)

    clf = {}
    print("\n=== clean ===")
    clf["clean"] = A.train_one("full", "clean", tr["target"], tr["y"], va["target"], va["y"],
                               Xva_m, mask_pool, args, dev, prep)
    print("\n=== mask_aware ===")
    clf["mask_aware"] = A.train_one("full", "mask_aware", tr["target"], tr["y"], va["target"], va["y"],
                                    Xva_m, mask_pool, args, dev, prep)
    print("\n=== fill_aware ===")
    clf["fill_aware"] = A.train_one("full", "clean", Xtr_f, tr["y"], Xva_f, va["y"],
                                    Xva_m, mask_pool, args, dev, prep)

    print("\nbuilding test inputs ...")
    cl, m = te["cloudy"], te["mask"]
    sets = {
        "cloud_free": te["target"],
        "cloudy": cl,
        "masked_zero": (cl * (1 - m[..., None])).astype(np.uint8),
    }
    sets.update(T.fill_all(cl, m, unet, dev))

    rows = []
    for bname, (lo, hi) in REAL_BINS.items():
        sel = (te["cov"] >= lo) & (te["cov"] < hi)
        n = int(sel.sum())
        if n == 0:
            continue
        for inp in INPUTS:
            for cname in CLFS:
                acc = float(A.accuracy(clf[cname], sets[inp][sel], te["y"][sel], prep, dev))
                rows.append(dict(seed=args.seed, classifier=cname, bin=bname, input=inp,
                                 acc=round(acc, 4), ci95=round(1.96 * np.sqrt(acc * (1 - acc) / n), 4), n=n))

    df = pd.DataFrame(rows)
    out = Path("cloud_removal_project/results")
    out.mkdir(parents=True, exist_ok=True)
    out_csv = out / f"sen12_threeway_seed{args.seed}.csv"
    df.to_csv(out_csv, index=False)

    pd.set_option("display.width", 220)
    for b in df.bin.unique():
        sub = df[df.bin == b]
        print(f"\n=== bin {b} (n={sub.n.iloc[0]}), seed {args.seed} ===")
        print(sub.pivot_table(index="classifier", columns="input", values="acc").round(3).to_string())
    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()
