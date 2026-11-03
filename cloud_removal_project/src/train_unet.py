"""
Phase 3: Training the Lightweight U-Net Inpainter on EuroSAT.
Trains on synthetic cloud-masked EuroSAT tiles with L1 reconstruction loss.
"""

import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np

from dataset import EuroSATDataset
from unet_inpaint import SmallUNetInpainter
from cloud_simulation import CloudSimulator

def train_unet(epochs=30, batch_size=64, lr=1e-3, device=None, max_train_samples=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training Small U-Net Inpainter on EuroSAT (device: {device}) ---")
    
    transform = T.Compose([
        T.ToTensor() # [0, 1]
    ])
    
    train_ds = EuroSATDataset("cloud_removal_project/data/splits/train.csv", transform=transform)
    if max_train_samples and max_train_samples < len(train_ds):
        train_ds.df = train_ds.df.sample(n=max_train_samples, random_state=42).reset_index(drop=True)
        print(f"Subsampled train set to {len(train_ds)} for fast CPU training.")
        
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_ds = EuroSATDataset("cloud_removal_project/data/splits/val.csv", transform=transform)
    val_ds.df = val_ds.df.sample(n=min(1000, len(val_ds.df)), random_state=0).reset_index(drop=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val_rng = np.random.default_rng(777)
    val_masks = [sim_v.generate_cloud_mask(val_rng.choice([(0.10, 0.30), (0.30, 0.60), (0.60, 0.80)]).tolist(), rng=val_rng)
                 for sim_v in [CloudSimulator(64, 64)] for _ in range(len(val_ds))]
    best_val = float("inf")
    
    model = SmallUNetInpainter().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    l1_loss = nn.L1Loss()
    
    sim = CloudSimulator(64, 64)
    rng = np.random.default_rng(42)
    
    save_path = Path("cloud_removal_project/models/unet_inpaint.pth")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        
        for batch_idx, (clean_imgs, _, _, _) in enumerate(loader):
            B, C, H, W = clean_imgs.shape
            
            # Generate random synthetic cloud masks for batch
            masks = []
            for _ in range(B):
                # randomly select coverage bin
                b = rng.choice([(0.10, 0.30), (0.30, 0.60), (0.60, 0.80)])
                m = sim.generate_cloud_mask(b, rng=rng)
                masks.append(m)
                
            masks_np = np.stack(masks, axis=0)[:, np.newaxis, :, :] # (B, 1, H, W)
            masks_t = torch.from_numpy(masks_np).float().to(device)
            
            # Base cloud replacement: white reflectance ~ 0.93 (238/255)
            cloud_noise = torch.normal(mean=0.93, std=0.03, size=(B, 3, H, W)).clamp(0.0, 1.0).to(device)
            cloudy_imgs = clean_imgs.to(device) * (1.0 - masks_t) + cloud_noise * masks_t
            
            optimizer.zero_grad()
            # Feed cloudy image and mask
            pred_composite = model(cloudy_imgs, masks_t)
            
            # Reconstruction loss: heavily weight the masked region
            loss_in_mask = l1_loss(pred_composite * masks_t, clean_imgs.to(device) * masks_t)
            loss_out_mask = l1_loss(pred_composite * (1.0 - masks_t), clean_imgs.to(device) * (1.0 - masks_t))
            loss = loss_in_mask * 5.0 + loss_out_mask
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if (batch_idx + 1) % 25 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx+1}/{len(loader)}] Inpaint Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(loader)
        elapsed = time.time() - t0
        # Validation (in-mask L1, fixed masks) -> keep best checkpoint
        model.eval()
        v_loss, v_n, k = 0.0, 0, 0
        with torch.no_grad():
            for v_imgs, _, _, _ in val_loader:
                B_v = v_imgs.size(0)
                m_t = torch.from_numpy(np.stack(val_masks[k:k + B_v])[:, None]).float().to(device)
                k += B_v
                v_imgs = v_imgs.to(device)
                noise = torch.full_like(v_imgs, 0.93)
                out = model(v_imgs * (1 - m_t) + noise * m_t, m_t)
                v_loss += (out - v_imgs).abs().mul(m_t).sum().item() / (m_t.sum().item() * 3 + 1e-8) * B_v
                v_n += B_v
        val_l1 = v_loss / v_n
        print(f"U-Net Epoch {epoch} complete in {elapsed:.1f}s | Avg Loss: {avg_loss:.4f} | Val in-mask L1: {val_l1:.4f}")
        if val_l1 < best_val:
            best_val = val_l1
            torch.save(model.state_dict(), save_path)
        
    print(f"Best val in-mask L1: {best_val:.4f}. Checkpoint (best) at {save_path}")
    return model

if __name__ == "__main__":
    train_unet()
