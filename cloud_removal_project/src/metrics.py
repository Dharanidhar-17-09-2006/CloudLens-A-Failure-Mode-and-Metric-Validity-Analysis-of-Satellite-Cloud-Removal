"""
Phase 2: Metrics computation module.
Computes PSNR, SSIM, and LPIPS inside the cloud mask and across the full image.
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch

class MetricEvaluator:
    def __init__(self, use_lpips=True, device="cpu"):
        self.device = device
        self.use_lpips = use_lpips
        self.lpips_model = None
        if self.use_lpips:
            try:
                import lpips
                self.lpips_model = lpips.LPIPS(net="alex").to(device)
                self.lpips_model.eval()
            except Exception as e:
                print(f"Warning: LPIPS could not be loaded: {e}. LPIPS will default to NaN.")

    def compute_psnr(self, gt, pred, mask=None):
        """
        gt, pred: uint8 or float [0, 255], HxWx3
        mask: optional binary mask (1 for cloud/inpaint region)
        """
        gt_f = gt.astype(np.float64)
        pred_f = pred.astype(np.float64)
        
        if mask is not None and mask.sum() > 0:
            mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=-1)
            diff = (gt_f - pred_f)[mask_3d == 1]
            mse = np.mean(diff ** 2)
        else:
            mse = np.mean((gt_f - pred_f) ** 2)
            
        if mse == 0:
            return 100.0
        return float(10.0 * np.log10((255.0 ** 2) / mse))

    def compute_ssim(self, gt, pred, mask=None):
        """
        gt, pred: uint8 HxWx3
        mask: optional binary mask
        """
        try:
            score, ssim_map = ssim(
                gt, pred,
                channel_axis=2,
                data_range=255,
                full=True
            )
            if mask is not None and mask.sum() > 0:
                mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=-1)
                mask_ssim = ssim_map[mask_3d == 1].mean()
                return float(score), float(mask_ssim)
            return float(score), float(score)
        except Exception:
            return 0.0, 0.0

    def compute_lpips(self, gt, pred):
        """
        gt, pred: uint8 HxWx3
        Returns full-image perceptual distance.
        """
        if self.lpips_model is None:
            return float("nan")
        try:
            with torch.no_grad():
                # Convert uint8 [0, 255] to torch tensor [-1, 1], shape 1x3xHxW
                t_gt = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
                t_pr = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
                dist = self.lpips_model(t_gt.to(self.device), t_pr.to(self.device))
                return float(dist.squeeze().item())
        except Exception:
            return float("nan")

    def evaluate_all(self, gt, pred, mask):
        """
        Returns dictionary of all metrics.
        """
        psnr_full = self.compute_psnr(gt, pred)
        psnr_mask = self.compute_psnr(gt, pred, mask=mask)
        ssim_full, ssim_mask = self.compute_ssim(gt, pred, mask=mask)
        lpips_score = self.compute_lpips(gt, pred)
        
        return {
            "psnr_full": round(psnr_full, 3),
            "psnr_mask": round(psnr_mask, 3),
            "ssim_full": round(ssim_full, 4),
            "ssim_mask": round(ssim_mask, 4),
            "lpips": round(lpips_score, 4) if not np.isnan(lpips_score) else None
        }
