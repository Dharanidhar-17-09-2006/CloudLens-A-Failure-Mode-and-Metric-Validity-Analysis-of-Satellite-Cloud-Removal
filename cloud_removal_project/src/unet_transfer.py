"""
Transfer-learning check for the U-Net fill (EuroSAT, thick synthetic clouds).

Three U-Nets with a ResNet-18 encoder. Each uses the same decoder, data, loss (5x in-cloud L1 + 1x out-of-cloud L1),
Adam lr 1e-3, batch 64, cloud-noise model, best-validation checkpoint and epoch budget as the scratch U-Net:
  unet_rand      : random-init encoder
  unet_imagenet  : ImageNet-init encoder
  unet_sat       : encoder+decoder pretrained on UNLABELLED EuroSAT TRAIN tiles by block inpainting
                   (random 8x8 blocks hidden, 10 epochs), then trained on the cloud task.
The scratch U-Net (SmallUNetInpainter, already trained) is the reference.

Every U-Net fills the same test tiles (same cloud seeds as the ablation). The fills are scored by the
fill-aware classifier, retrained here with the threeway recipe (the threeway run did not save its models).
Only TRAIN tiles are used for any training. Test tiles are used only for scoring.

Writes: results/unet_transfer_seed{seed}.csv
Run:    python cloud_removal_project/src/unet_transfer.py --seed 0
"""
import argparse
import copy
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import ablate_classifier_finetune as A
import threeway_comparison as T
from cloud_simulation import CloudSimulator
from unet_inpaint import SmallUNetInpainter

TRAIN_BINS = [(0.10, 0.30), (0.30, 0.60), (0.60, 0.80)]


def resnet_encoder(in_ch, pretrained):
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = torchvision.models.resnet18(weights=weights)
    if in_ch != 3:
        old = m.conv1.weight.data
        new = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, :3] = old          # RGB weights kept; mask channel starts at zero
        m.conv1 = new
    return m


def up(x, ref):
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    )


class ResUNet(nn.Module):
    """ResNet-18 encoder U-Net. Input: masked RGB (3ch) + mask (1ch). Output: composite RGB in [0,1]."""

    def __init__(self, pretrained=False):
        super().__init__()
        self.enc = resnet_encoder(4, pretrained)
        self.dec4 = block(512 + 256, 256)
        self.dec3 = block(256 + 128, 128)
        self.dec2 = block(128 + 64, 64)
        self.dec1 = block(64 + 64, 64)
        self.dec0 = block(64, 32)
        self.out = nn.Sequential(nn.Conv2d(32, 3, 1), nn.Sigmoid())

    def forward(self, x_masked, mask):
        e = self.enc
        x = torch.cat([x_masked, mask], 1)
        x0 = e.relu(e.bn1(e.conv1(x)))      # 32x32
        x1 = e.layer1(e.maxpool(x0))        # 16x16
        x2 = e.layer2(x1)                   # 8x8
        x3 = e.layer3(x2)                   # 4x4
        x4 = e.layer4(x3)                   # 2x2
        d = self.dec4(torch.cat([up(x4, x3), x3], 1))
        d = self.dec3(torch.cat([up(d, x2), x2], 1))
        d = self.dec2(torch.cat([up(d, x1), x1], 1))
        d = self.dec1(torch.cat([up(d, x0), x0], 1))
        d = self.dec0(up(d, x_masked))
        return self.out(d)


def cloud_batch(sim, rng):
    """Same cloud model as train_unet.py: white cloud (0.93 +- 0.03) inside random cloud masks."""
    def f(clean):
        B = clean.shape[0]
        ms = [sim.generate_cloud_mask(TRAIN_BINS[rng.integers(3)], rng=rng) for _ in range(B)]
        m = torch.from_numpy(np.stack(ms)[:, None]).float().to(clean.device)
        noise = torch.normal(0.93, 0.03, size=clean.shape, device=clean.device).clamp(0, 1)
        return clean * (1 - m) + noise * m, m
    return f


def block_batch(rng, cell=8, p=0.5):
    """Self-supervised task on clear satellite tiles: hide random 8x8 blocks."""
    def f(clean):
        B, _, H, W = clean.shape
        g = (rng.random((B, 1, H // cell, W // cell)) < p).astype(np.float32)
        m = torch.from_numpy(np.kron(g, np.ones((1, 1, cell, cell), np.float32))).to(clean.device)
        return clean * (1 - m), m
    return f


@torch.no_grad()
def val_l1(model, vin, vm, vt, bs=256):
    model.eval()
    tot, denom = 0.0, 0.0
    for i in range(0, len(vin), bs):
        out = model(vin[i:i + bs], vm[i:i + bs])
        m = vm[i:i + bs]
        tot += ((out - vt[i:i + bs]).abs() * m).sum().item()
        denom += m.sum().item() * 3
    return tot / (denom + 1e-8)


def train_model(model, Xtr, make_batch, epochs, bs, lr, dev, rng, tag, val=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    l1 = nn.L1Loss()
    N = len(Xtr)
    best_v, best_state = np.inf, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(N)
        t0, tot, nb = time.time(), 0.0, 0
        for i in range(0, N, bs):
            idx = np.sort(perm[i:i + bs])
            clean = torch.from_numpy(Xtr[idx]).to(dev).permute(0, 3, 1, 2).float() / 255.0
            inp, m = make_batch(clean)
            out = model(inp, m)
            loss = l1(out * m, clean * m) * 5.0 + l1(out * (1 - m), clean * (1 - m))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item()
            nb += 1
        msg = f"[{tag}] epoch {ep}/{epochs} loss {tot / nb:.4f} ({time.time() - t0:.0f}s)"
        if val is not None:
            v = val_l1(model, *val)
            msg += f" | val in-mask L1 {v:.4f}"
            if v < best_v:
                best_v, best_state = v, copy.deepcopy(model.state_dict())
        print(msg)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def unet_fill(model, cloudy, masks, dev, bs=256):
    model.eval()
    out = []
    for i in range(0, len(cloudy), bs):
        x = torch.from_numpy(cloudy[i:i + bs]).to(dev).permute(0, 3, 1, 2).float() / 255.0
        m = torch.from_numpy(masks[i:i + bs]).to(dev).unsqueeze(1).float()
        out.append((model(x, m).permute(0, 2, 3, 1).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--pre_epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--imagenet_lr", type=float, default=None,
                    help="control run: also train the ImageNet U-Net at this learning rate (separate checkpoint)")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}  seed: {args.seed}")
    torch.manual_seed(args.seed)

    sp = "cloud_removal_project/data/splits"
    Xtr, ytr = A.load_split(f"{sp}/train.csv", None)
    Xva, yva = A.load_split(f"{sp}/val.csv", 2000)
    Xte, yte = A.load_split(f"{sp}/test.csv", None)
    print(f"train {len(Xtr)}  val {len(Xva)}  test {len(Xte)}")

    sim = CloudSimulator(64, 64)
    prep = A.make_prep(dev)

    scratch = SmallUNetInpainter().to(dev)
    scratch.load_state_dict(torch.load("cloud_removal_project/models/unet_inpaint.pth", map_location=dev))
    scratch.eval()

    # Fill-aware classifier, same recipe as threeway_comparison.py.
    cargs = types.SimpleNamespace(epochs=3, bs=128, seed=args.seed, scratch=False)
    print("training fill-aware classifier ...")
    Xtr_f = T.build_fill_set(Xtr, sim, scratch, dev, np.random.default_rng(args.seed), clean_frac=0.2)
    Xva_f = T.build_fill_set(Xva, sim, scratch, dev, np.random.default_rng(777), clean_frac=0.0)
    clf = A.train_one("full", "clean", Xtr_f, ytr, Xva_f, yva, Xva, None, cargs, dev, prep)
    del Xtr_f, Xva_f

    # Fixed validation set for checkpoint selection (cloud task), same idea as train_unet.py.
    vr = np.random.default_rng(99)
    vm_np, vc_np = [], []
    torch.manual_seed(99)
    for img in Xva:
        m = sim.generate_cloud_mask(TRAIN_BINS[vr.integers(3)], rng=vr)
        vm_np.append(m)
        vc_np.append(img)
    vt = torch.from_numpy(Xva).to(dev).permute(0, 3, 1, 2).float() / 255.0
    vm = torch.from_numpy(np.stack(vm_np)[:, None]).float().to(dev)
    noise = torch.normal(0.93, 0.03, size=vt.shape, device=dev).clamp(0, 1)
    vin = vt * (1 - vm) + noise * vm
    val = (vin, vm, vt)

    variants = {}
    models_dir = Path("cloud_removal_project/models")
    models_dir.mkdir(parents=True, exist_ok=True)

    def get_or_train(name, pretrained, make_batch, rng_seed, init_state=None, lr=None):
        """Train one U-Net and save it. If its checkpoint exists, load it instead of retraining."""
        path = models_dir / f"unet_transfer_{name}_seed{args.seed}.pth"
        m = ResUNet(pretrained=pretrained).to(dev)
        if path.exists():
            print(f"\n=== {name}: checkpoint found, skipping training ===")
            m.load_state_dict(torch.load(path, map_location=dev))
            return m
        print(f"\n=== {name} ===")
        if init_state is not None:
            m.load_state_dict(init_state)
        rng = np.random.default_rng(rng_seed)
        lr_use = args.lr if lr is None else lr
        m = train_model(m, Xtr, make_batch(rng), args.epochs, args.bs, lr_use, dev, rng, name, val)
        torch.save(m.state_dict(), path)
        print(f"saved {path}")
        return m

    pre_path = models_dir / f"unet_transfer_sat_pretrain_seed{args.seed}.pth"
    if pre_path.exists():
        print("\n=== satellite pretraining: checkpoint found, skipping ===")
        pre_state = torch.load(pre_path, map_location=dev)
    else:
        print("\n=== satellite pretraining (block inpainting, unlabelled train tiles) ===")
        pm = ResUNet(pretrained=False).to(dev)
        rng = np.random.default_rng(args.seed + 3)
        pm = train_model(pm, Xtr, block_batch(rng), args.pre_epochs, args.bs, args.lr, dev,
                         rng, "sat-pretrain", None)
        torch.save(pm.state_dict(), pre_path)
        pre_state = pm.state_dict()

    variants["unet_rand"] = get_or_train("unet_rand", False, lambda r: cloud_batch(sim, r), args.seed + 1)
    variants["unet_imagenet"] = get_or_train("unet_imagenet", True, lambda r: cloud_batch(sim, r), args.seed + 2)
    variants["unet_sat"] = get_or_train("unet_sat", False, lambda r: cloud_batch(sim, r), args.seed + 4,
                                        init_state=pre_state)
    suffix = ""
    if args.imagenet_lr is not None:
        name = f"unet_imagenet_lr{args.imagenet_lr:g}"
        variants[name] = get_or_train(name, True, lambda r: cloud_batch(sim, r), args.seed + 2,
                                      lr=args.imagenet_lr)
        suffix = f"_imagenet_lr{args.imagenet_lr:g}"
    variants["unet_scratch_ref"] = scratch

    print("\nscoring test tiles ...")
    rows = []
    for bi, (bname, brange) in enumerate(A.BINS.items()):
        masks, cloudy = [], []
        for idx, img in enumerate(Xte):
            r = np.random.default_rng(42 + idx * 10 + bi)
            mk = sim.generate_cloud_mask(brange, rng=r)
            c, _ = sim.apply_cloud(img, mk, condition="thick", rng=r)
            masks.append(mk)
            cloudy.append(c)
        masks, cloudy = np.stack(masks), np.stack(cloudy)

        fills = {k: unet_fill(v, cloudy, masks, dev) for k, v in variants.items()}
        ref = T.fill_all(cloudy, masks, scratch, dev)
        fills["telea_ref"] = ref["telea"]
        fills["nearest_neighbor_ref"] = ref["nearest_neighbor"]

        for name, imgs in fills.items():
            acc = float(A.accuracy(clf, imgs, yte, prep, dev))
            n = len(yte)
            rows.append(dict(seed=args.seed, bin=bname, method=name, acc=round(acc, 4),
                             ci95=round(1.96 * np.sqrt(acc * (1 - acc) / n), 4), n=n))
        print(f"bin {bname} done")

    df = pd.DataFrame(rows)
    out = Path("cloud_removal_project/results")
    out.mkdir(parents=True, exist_ok=True)
    out_csv = out / f"unet_transfer_seed{args.seed}{suffix}.csv"
    df.to_csv(out_csv, index=False)
    pd.set_option("display.width", 200)
    print("\n=== accuracy, fill-aware classifier ===")
    print(df.pivot_table(index="method", columns="bin", values="acc").round(3).to_string())
    print(f"\nSaved to {out_csv}")


if __name__ == "__main__":
    main()
