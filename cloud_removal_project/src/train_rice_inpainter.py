"""
Train an inpainter on RICE: clear tiles + REAL cloud masks (self-supervised, no labels).
Held-out images (val/test splits) are never used for training.

  python cloud_removal_project/src/train_rice_inpainter.py --name small --arch small
  python cloud_removal_project/src/train_rice_inpainter.py --name big --arch big
  python cloud_removal_project/src/train_rice_inpainter.py --name big_gan --arch big --gan 1
Checkpoints: cloud_removal_project/models/rice_<name>.pth (best validation in-mask L1)
"""
import argparse, time
import numpy as np, torch
import torch.nn.functional as F
import rice_data as D
from unet_inpaint import SmallUNetInpainter
from rice_models import BigUNetInpainter, PatchDiscriminator


def to_t(a, device):
    return torch.from_numpy(a).permute(0, 3, 1, 2).float().div(255).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--arch", choices=["small", "big"], default="big")
    ap.add_argument("--gan", type=int, default=0); ap.add_argument("--iters", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lpips_w", type=float, default=0.1); ap.add_argument("--adv_w", type=float, default=0.05)
    ap.add_argument("--eval_every", type=int, default=500)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    torch.manual_seed(0)

    rice1, rice2, splits = D.get_data()
    sampler = D.TrainSampler(rice1, rice2, splits, seed=0)
    val = D.eval_tiles(rice2, splits["RICE2"]["val"])[:400]
    vx = to_t(np.stack([t["clear"] for t in val]), device)
    vm = torch.from_numpy(np.stack([t["mask"] for t in val])).unsqueeze(1).float().to(device)
    print(f"{len(val)} validation tiles")

    G = (BigUNetInpainter(base=48) if a.arch == "big" else SmallUNetInpainter()).to(device)
    print(f"generator params: {sum(p.numel() for p in G.parameters()) / 1e6:.2f}M")
    optG = torch.optim.Adam(G.parameters(), lr=a.lr, betas=(0.5, 0.999) if a.gan else (0.9, 0.999))
    if a.gan:
        Dnet = PatchDiscriminator().to(device)
        optD = torch.optim.Adam(Dnet.parameters(), lr=a.lr, betas=(0.5, 0.999))
    lp = None
    if a.lpips_w > 0:
        import lpips
        lp = lpips.LPIPS(net="alex").to(device).eval()
        for p in lp.parameters():
            p.requires_grad = False

    ckpt = D.R.ROOT / "models" / f"rice_{a.name}.pth"
    ckpt.parent.mkdir(exist_ok=True)
    best, t0 = float("inf"), time.time()
    for it in range(1, a.iters + 1):
        G.train()
        xb, mb = sampler.batch(a.bs)
        y = to_t(xb, device)
        m = torch.from_numpy(mb).unsqueeze(1).float().to(device)
        hole = torch.normal(0.93, 0.03, size=y.shape, device=device).clamp(0, 1)   # same hole value as the EuroSAT U-Net
        out = G(y * (1 - m) + hole * m, m)

        if a.gan:
            optD.zero_grad()
            d_loss = F.relu(1 - Dnet(y, m)).mean() + F.relu(1 + Dnet(out.detach(), m)).mean()
            d_loss.backward(); optD.step()

        l1_in = ((out - y).abs() * m).sum() / (m.sum() * 3 + 1e-8)
        l1_out = ((out - y).abs() * (1 - m)).sum() / ((1 - m).sum() * 3 + 1e-8)
        loss = 5.0 * l1_in + l1_out
        if lp is not None:
            loss = loss + a.lpips_w * lp(out * 2 - 1, y * 2 - 1).mean()
        if a.gan:
            loss = loss + a.adv_w * (-Dnet(out, m).mean())
        optG.zero_grad(); loss.backward(); optG.step()

        if it % a.eval_every == 0 or it == a.iters:
            G.eval()
            with torch.no_grad():
                hv = torch.full_like(vx, 0.93)
                o = G(vx * (1 - vm) + hv * vm, vm)
                v = (((o - vx).abs() * vm).sum() / (vm.sum() * 3)).item()
            flag = ""
            if v < best:
                best = v; torch.save(G.state_dict(), ckpt); flag = "  <- saved"
            print(f"iter {it}/{a.iters} | loss {loss.item():.4f} | val in-mask L1 {v:.4f} | {time.time() - t0:.0f}s{flag}")
    print(f"best val in-mask L1 {best:.4f}; checkpoint {ckpt}")

if __name__ == "__main__":
    main()
