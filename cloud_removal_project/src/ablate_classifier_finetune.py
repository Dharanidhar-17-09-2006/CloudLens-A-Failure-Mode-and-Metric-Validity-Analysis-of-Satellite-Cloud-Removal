"""
Fine-tuning ablation for the downstream land-cover classifier.

Question: is adapting the classifier (unfreezing more layers, training on masked tiles)
better than reconstructing the cloud first?

Regimes (all start from ImageNet ResNet-18): head_only, layer4, full.
Training modes: clean (no clouds) and mask_aware (80% of samples get random shape masks).
Evaluation (EuroSAT test subset, thick synthetic clouds from the held-out cloud-mask
family, per coverage bin): clean, cloudy, masked_zero, mean_fill, telea.

Run:  python cloud_removal_project/src/ablate_classifier_finetune.py
Quick test:  python cloud_removal_project/src/ablate_classifier_finetune.py --epochs 1 --max_train 3000 --n_test 500
"""
import argparse, time
from pathlib import Path
import numpy as np, pandas as pd, torch, torch.nn as nn
from PIL import Image

from dataset import EuroSATDataset
from classifier import get_classifier, NORM_MEAN, NORM_STD
from cloud_simulation import CloudSimulator
from inpainting_baselines import masked_input, mean_fill, telea_inpaint

BINS = {"10-30%": (0.10, 0.30), "30-60%": (0.30, 0.60), "60-80%": (0.60, 0.80)}
REGIMES = {
    "head_only": dict(train=["fc"], lr=1e-3),
    "layer4": dict(train=["layer4", "fc"], lr=1e-3),
    "full": dict(train=None, lr=3e-4),
}
INPUTS = ["clean", "cloudy", "masked_zero", "mean_fill", "telea"]


def load_split(csv, n=None, seed=0):
    ds = EuroSATDataset(csv)
    if n and n < len(ds):
        ds.df = ds.df.sample(n=n, random_state=seed).reset_index(drop=True)
    X = np.stack([np.array(Image.open(ds.base_dir / r).convert("RGB")) for r in ds.df["rel_path"]])
    return X, ds.df["label"].astype(int).values


def build_eval_sets(X, sim):
    """Per bin: dict input_type -> uint8 array. Seeds differ from classifier-training mask seeds."""
    sets = {}
    for bi, (bname, brange) in enumerate(BINS.items()):
        d = {k: [] for k in INPUTS}
        for idx, img in enumerate(X):
            rng = np.random.default_rng(42 + idx * 10 + bi)
            mask = sim.generate_cloud_mask(brange, rng=rng)
            cloudy, _ = sim.apply_cloud(img, mask, condition="thick", rng=rng)
            d["clean"].append(img)
            d["cloudy"].append(cloudy)
            d["masked_zero"].append(masked_input(cloudy, mask, 0))
            d["mean_fill"].append(mean_fill(cloudy, mask))
            d["telea"].append(telea_inpaint(cloudy, mask))
        sets[bname] = {k: np.stack(v) for k, v in d.items()}
    return sets


def make_prep(dev):
    mean = torch.tensor(NORM_MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(NORM_STD, device=dev).view(1, 3, 1, 1)
    return lambda x01: (x01 - mean) / std


@torch.no_grad()
def accuracy(model, imgs, y, prep, dev, bs=256):
    model.eval()
    correct = 0
    for i in range(0, len(imgs), bs):
        xb = torch.from_numpy(imgs[i:i + bs]).to(dev).permute(0, 3, 1, 2).float() / 255.0
        pred = model(prep(xb)).argmax(1).cpu().numpy()
        correct += (pred == y[i:i + bs]).sum()
    return correct / len(imgs)


def set_train_mode(model, spec):
    model.eval()  # frozen parts keep fixed BatchNorm statistics
    if spec["train"] is None:
        model.train()
    else:
        for name in spec["train"]:
            getattr(model, name).train()


def train_one(regime, mode, Xtr, ytr, Xva, yva, mask_pool, args, dev, prep):
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    spec = REGIMES[regime]
    model = get_classifier(10, pretrained=not args.scratch).to(dev)
    for n, p in model.named_parameters():
        p.requires_grad = spec["train"] is None or any(n.startswith(t) for t in spec["train"])
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=spec["lr"])
    crit = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(Xtr).to(dev)
    yt = torch.from_numpy(ytr).long().to(dev)
    N = len(Xt)
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        set_train_mode(model, spec)
        perm = rng.permutation(N)
        for i in range(0, N, args.bs):
            idx = torch.from_numpy(perm[i:i + args.bs]).to(dev)
            xb = Xt[idx].permute(0, 3, 1, 2).float() / 255.0
            B = xb.size(0)
            xb = torch.where((torch.rand(B, device=dev) < 0.5).view(-1, 1, 1, 1), xb.flip(3), xb)
            xb = torch.where((torch.rand(B, device=dev) < 0.5).view(-1, 1, 1, 1), xb.flip(2), xb)
            if mode == "mask_aware":
                ms = []
                for _ in range(B):
                    if rng.random() < 0.8:
                        pool = mask_pool[list(BINS)[rng.integers(3)]]
                        ms.append(pool[rng.integers(len(pool))])
                    else:
                        ms.append(np.zeros((64, 64), np.uint8))
                m = torch.from_numpy(np.stack(ms)[:, None]).float().to(dev)
                xb = xb * (1.0 - m)  # black fill, same as masked_zero at test time
            opt.zero_grad()
            loss = crit(model(prep(xb)), yt[idx])
            loss.backward()
            opt.step()
        va = accuracy(model, Xva, yva, prep, dev)
        print(f"  [{regime}/{mode}] epoch {ep}/{args.epochs} ({time.time() - t0:.0f}s) clean val acc {va:.4f}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--n_test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scratch", action="store_true", help="no ImageNet weights (smoke test only)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")

    sp = "cloud_removal_project/data/splits"
    Xtr, ytr = load_split(f"{sp}/train.csv", args.max_train)
    Xva, yva = load_split(f"{sp}/val.csv", 2000)
    Xte, yte = load_split(f"{sp}/test.csv", args.n_test)
    print(f"train {len(Xtr)}  val {len(Xva)}  test {len(Xte)}")

    sim = CloudSimulator(64, 64)
    mrng = np.random.default_rng(1234)  # held-out shape-mask family, as in the existing mask-aware classifier
    mask_pool = {b: np.stack([sim.generate_shape_mask(r, rng=mrng) for _ in range(1500)]) for b, r in BINS.items()}
    eval_sets = build_eval_sets(Xte, sim)
    prep = make_prep(dev)

    rows = []
    for regime in REGIMES:
        for mode in ["clean", "mask_aware"]:
            print(f"\n=== {regime} / {mode} ===")
            model = train_one(regime, mode, Xtr, ytr, Xva, yva, mask_pool, args, dev, prep)
            for bname in BINS:
                for inp in INPUTS:
                    if inp == "clean" and bname != "10-30%":
                        continue
                    acc = accuracy(model, eval_sets[bname][inp], yte, prep, dev)
                    rows.append(dict(regime=regime, train_mode=mode, bin="all" if inp == "clean" else bname,
                                     input=inp, acc=round(float(acc), 4), n=len(yte)))
            del model
            if dev == "cuda":
                torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    out = Path("cloud_removal_project/results"); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "classifier_ablation.csv", index=False)
    pd.set_option("display.width", 200)
    for b in ["all"] + list(BINS):
        sub = df[df["bin"] == b]
        print(f"\n=== accuracy, coverage bin: {b} ===")
        print(sub.pivot_table(index=["regime", "train_mode"], columns="input", values="acc").round(3).to_string())
    print(f"\nSaved to {out / 'classifier_ablation.csv'}")


if __name__ == "__main__":
    main()
