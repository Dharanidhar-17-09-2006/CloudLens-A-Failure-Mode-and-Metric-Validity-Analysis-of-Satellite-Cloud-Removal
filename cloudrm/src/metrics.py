import numpy as np
import torch
import lpips
from skimage.metrics import structural_similarity

_lp = None


def psnr(a, b):
    mse = float(np.mean((a - b) ** 2))
    return 100.0 if mse == 0 else 10 * np.log10(1.0 / mse)


def ssim(a, b):
    return float(structural_similarity(a, b, channel_axis=-1, data_range=1.0))


def lpips_score(a, b):
    global _lp
    if _lp is None:
        _lp = lpips.LPIPS(net="alex")
    t = lambda x: torch.from_numpy(x.transpose(2, 0, 1).copy())[None] * 2 - 1
    with torch.no_grad():
        return float(_lp(t(a), t(b)))


def sam(a, b, eps=1e-8):
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + eps
    return float(np.degrees(np.arccos(np.clip(num / den, -1, 1))).mean())


def ndvi(red, nir, eps=1e-8):
    return (nir - red) / (nir + red + eps)


def ndvi_err(pred_red, pred_nir, ref_red, ref_nir):
    return float(np.abs(ndvi(pred_red, pred_nir) - ndvi(ref_red, ref_nir)).mean())
