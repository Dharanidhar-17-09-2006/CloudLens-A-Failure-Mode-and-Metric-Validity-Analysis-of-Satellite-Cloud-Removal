"""
Fill-Aware Classifier: ResNet-18 fine-tuned on FILLED images.

Fair counterpart to the mask-aware classifier. Both are adapted to cloud gaps and both
train on the held-out shape-mask family (never the test cloud generator), so the
comparison "adapt the classifier to masks" vs "adapt it to fills" is like for like.

Each training sample: random mask -> thick cloud -> random fill method
(mean / nearest-neighbor / Telea, plus U-Net if a checkpoint exists) -> classifier.
"""

import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image

from dataset import EuroSATDataset
from classifier import get_classifier, NORM_MEAN, NORM_STD
from cloud_simulation import CloudSimulator
import inpainting_baselines as baselines

BINS = [(0.10, 0.30), (0.30, 0.60), (0.60, 0.80)]


def _load_unet(device):
    from unet_inpaint import SmallUNetInpainter
    ckpt = Path("cloud_removal_project/models/unet_inpaint.pth")
    if not ckpt.exists():
        return None
    net = SmallUNetInpainter().to(device)
    net.load_state_dict(torch.load(ckpt, map_location=device))
    net.eval()
    return net


def train_fill_classifier(epochs=8, batch_size=64, lr=5e-4, device=None, max_train_samples=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training Fill-Aware ResNet-18 Classifier (device: {device}) ---")

    norm = T.Compose([T.ToTensor(), T.Normalize(mean=NORM_MEAN, std=NORM_STD)])
    # dataset returns PIL (no transform); we corrupt/fill in numpy, then normalize
    train_ds = EuroSATDataset("cloud_removal_project/data/splits/train.csv", transform=None)
    val_ds = EuroSATDataset("cloud_removal_project/data/splits/val.csv", transform=None)
    if max_train_samples and max_train_samples < len(train_ds):
        train_ds.df = train_ds.df.sample(n=max_train_samples, random_state=42).reset_index(drop=True)

    collate = lambda b: ([x[0] for x in b], torch.tensor([x[1] for x in b]))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)

    model = get_classifier(num_classes=10, pretrained=False)
    clean_ckpt = Path("cloud_removal_project/models/resnet18_clean.pth")
    if clean_ckpt.exists():
        model.load_state_dict(torch.load(clean_ckpt, map_location=device))
        print("Initialized from clean ResNet-18 weights.")
    else:
        model = get_classifier(num_classes=10, pretrained=True)
        print("Clean checkpoint not found; starting from ImageNet weights.")
    model.to(device)

    for p in model.parameters():
        p.requires_grad = False
    for module in (model.layer3, model.layer4, model.fc):
        for p in module.parameters():
            p.requires_grad = True

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    sim = CloudSimulator(64, 64)
    unet = _load_unet(device)
    fills = ["mean", "nn", "telea"] + (["unet"] if unet is not None else [])
    print(f"Training fills: {fills}")

    def make_filled(pil_img, rng, fixed_bin=None, fixed_fill=None, shape_masks=True):
        rgb = np.array(pil_img.convert("RGB"))
        b = fixed_bin or BINS[rng.integers(0, len(BINS))]
        mask = sim.generate_shape_mask(b, rng=rng) if shape_masks else sim.generate_cloud_mask(b, rng=rng)
        cloudy, _ = sim.apply_cloud(rgb, mask, condition="thick", rng=rng)
        f = fixed_fill or fills[rng.integers(0, len(fills))]
        if f == "mean":
            out = baselines.mean_fill(cloudy, mask)
        elif f == "nn":
            out = baselines.nearest_neighbor_fill(cloudy, mask)
        elif f == "telea":
            out = baselines.telea_inpaint(cloudy, mask)
        else:
            with torch.no_grad():
                ti = torch.from_numpy(cloudy).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(device)
                tm = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float().to(device)
                o = unet(ti, tm).squeeze(0).permute(1, 2, 0).cpu().numpy()
            out = (o * 255.0).clip(0, 255).astype(np.uint8)
        return out

    rng = np.random.default_rng(4321)          # differs from test seeds (42+idx)
    val_rng = np.random.default_rng(999)
    save_path = Path("cloud_removal_project/models/resnet18_fill_aware.pth")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    best = 0.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        correct = total = 0
        for pil_imgs, labels in train_loader:
            batch = []
            for im in pil_imgs:
                if rng.random() < 0.15:                      # keep some clean images
                    out = np.array(im.convert("RGB"))
                    if rng.random() < 0.5:
                        out = np.ascontiguousarray(out[:, ::-1])
                else:
                    out = make_filled(im, rng)
                batch.append(norm(Image.fromarray(out)))
            x = torch.stack(batch).to(device)
            y = labels.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        # validation on filled val images (30-60% bin, held-out shape masks, fixed rng)
        model.eval()
        vc = vt = 0
        with torch.no_grad():
            for pil_imgs, labels in val_loader:
                x = torch.stack([norm(Image.fromarray(make_filled(im, val_rng, fixed_bin=(0.30, 0.60))))
                                 for im in pil_imgs]).to(device)
                vc += (model(x).argmax(1).cpu() == labels).sum().item()
                vt += labels.size(0)
        val_acc = vc / vt
        print(f"Fill-Aware Epoch [{epoch}/{epochs}] ({time.time() - t0:.1f}s) | "
              f"Train Acc: {correct / total:.4f} | Filled Val Acc: {val_acc:.4f}")
        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best Fill-Aware checkpoint ({val_acc:.4f})")
    print(f"Done. Checkpoint at {save_path}")
    return model


if __name__ == "__main__":
    train_fill_classifier()
