"""
Supervised cloud removal on REAL PAIRS (RICE): input = the real cloudy tile (cloud pixels NOT replaced) + mask,
target = the paired clear tile. Unlike the self-supervised inpainters, this model sees the real cloud
appearance (haze rims, thin cloud that still carries signal) and can learn to use or correct it.
Pixels outside the mask are kept as in the input (composite), so with a dilated/estimated mask it behaves
like "restore only the cloudy region".

  python cloud_removal_project/src/train_paired_restorer.py --name small --arch small
  python cloud_removal_project/src/train_paired_restorer.py --name big --arch big
Checkpoints: cloud_removal_project/models/rice_paired_<name>.pth
"""
import argparse, time
import numpy as np, torch
import rice_data as D
from unet_inpaint import SmallUNetInpainter
from rice_models import BigUNetInpainter


class PairedSampler:
    def __init__(self, rice1, rice2, splits, tile=64, seed=0):
        self.tile, self.rng = tile, np.random.default_rng(seed)
        self.items = [rice1[k] for k in splits["RICE1"]["train"]] + [rice2[k] for k in splits["RICE2"]["train"]]

    def one(self):
        d = self.items[self.rng.integers(len(self.items))]
        t = self.tile
        for _ in range(50):
            y, x = self.rng.integers(0, D.SIZE - t + 1, 2)
            m = d["mask"][y:y + t, x:x + t]
            if 0.10 <= m.mean() <= 0.80:
                break
        a, b, c = d["cloud"][y:y + t, x:x + t], d["clear"][y:y + t, x:x + t], m
        k, flip = int(self.rng.integers(0, 4)), self.rng.random() < 0.5
        f = self.rng.uniform(0.9, 1.1)
        out = []
        for arr in (a, b, c):
            if flip:
                arr = arr[:, ::-1]
            out.append(np.ascontiguousarray(np.rot90(arr, k)))
        a, b, c = out
        a = np.clip(a.astype(np.float32) * f, 0, 255).astype(np.uint8)
        b = np.clip(b.astype(np.float32) * f, 0, 255).astype(np.uint8)
        return a, b, c

    def batch(self, B):
        s = [self.one() for _ in range(B)]
        return tuple(np.stack([p[i] for p in s]) for i in range(3))


def to_t(a, device):
    return torch.from_numpy(a).permute(0, 3, 1, 2).float().div(255).to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--arch", choices=["small", "big"], default="big")
    ap.add_argument("--iters", type=int, default=8000); ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4); ap.add_argument("--lpips_w", type=float, default=0.1)
    ap.add_argument("--eval_every", type=int, default=500)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    rice1, rice2, splits = D.get_data()
    sampler = PairedSampler(rice1, rice2, splits)
    val = D.eval_tiles(rice2, splits["RICE2"]["val"])[:300] + D.eval_tiles(rice1, splits["RICE1"]["val"])[:200]
    vx = to_t(np.stack([t["cloudy"] for t in val]), device)
    vy = to_t(np.stack([t["clear"] for t in val]), device)
    vm = torch.from_numpy(np.stack([t["mask"] for t in val])).unsqueeze(1).float().to(device)
    print(f"{len(val)} validation tiles (real cloudy + real mask)")

    G = (BigUNetInpainter(base=48) if a.arch == "big" else SmallUNetInpainter()).to(device)
    opt = torch.optim.Adam(G.parameters(), lr=a.lr)
    import lpips
    lp = lpips.LPIPS(net="alex").to(device).eval()
    for p in lp.parameters():
        p.requires_grad = False
    ckpt = D.R.ROOT / "models" / f"rice_paired_{a.name}.pth"
    best, t0 = float("inf"), time.time()
    for it in range(1, a.iters + 1):
        G.train()
        xb, yb, mb = sampler.batch(a.bs)
        x, y = to_t(xb, device), to_t(yb, device)
        m = torch.from_numpy(mb).unsqueeze(1).float().to(device)
        out = G(x, m)
        l1_in = ((out - y).abs() * m).sum() / (m.sum() * 3 + 1e-8)
        loss = 5.0 * l1_in + a.lpips_w * lp(out * 2 - 1, y * 2 - 1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if it % a.eval_every == 0 or it == a.iters:
            G.eval()
            with torch.no_grad():
                o = G(vx, vm)
                v = (((o - vy).abs() * vm).sum() / (vm.sum() * 3)).item()
            flag = ""
            if v < best:
                best = v; torch.save(G.state_dict(), ckpt); flag = "  <- saved"
            print(f"iter {it}/{a.iters} | loss {loss.item():.4f} | val in-mask L1 {v:.4f} | {time.time() - t0:.0f}s{flag}")
    print(f"best val in-mask L1 {best:.4f}; checkpoint {ckpt}")

if __name__ == "__main__":
    main()
