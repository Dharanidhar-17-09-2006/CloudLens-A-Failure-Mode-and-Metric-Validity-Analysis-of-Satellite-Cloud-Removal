"""
Fair Masked Baseline: Mask-Aware Downstream Classifier.
Fine-tunes ResNet-18 with random cloud masks applied during training,
teaching the model to classify partially obscured land-cover tiles without hallucinating.
"""

import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np

from dataset import EuroSATDataset
from classifier import get_classifier, NORM_MEAN, NORM_STD
from cloud_simulation import CloudSimulator

def train_mask_aware_classifier(epochs=8, batch_size=64, lr=5e-4, device=None, max_train_samples=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training Mask-Aware ResNet-18 Classifier (device: {device}) ---")
    
    # Base transforms
    transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    
    train_ds = EuroSATDataset("cloud_removal_project/data/splits/train.csv", transform=transform)
    val_ds = EuroSATDataset("cloud_removal_project/data/splits/val.csv", transform=transform)
    
    if max_train_samples and max_train_samples < len(train_ds):
        train_ds.df = train_ds.df.sample(n=max_train_samples, random_state=42).reset_index(drop=True)
        print(f"Subsampled train set to {len(train_ds)} for efficient CPU fine-tuning.")
        
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Initialize from the clean classifier weights
    model = get_classifier(num_classes=10, pretrained=False)
    clean_ckpt = Path("cloud_removal_project/models/resnet18_clean.pth")
    if clean_ckpt.exists():
        model.load_state_dict(torch.load(clean_ckpt, map_location=device))
        print("Initialized from clean ResNet-18 weights.")
    else:
        print("Clean checkpoint not found; starting from ImageNet weights.")
        model = get_classifier(num_classes=10, pretrained=True)
        
    model.to(device)
    
    # Train top layers (layer3, layer4, fc)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.layer3.parameters():
        p.requires_grad = True
    for p in model.layer4.parameters():
        p.requires_grad = True
    for p in model.fc.parameters():
        p.requires_grad = True
        
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    sim = CloudSimulator(64, 64)
    rng = np.random.default_rng(1234)  # differs from test seeds (42+idx)
    
    save_path = Path("cloud_removal_project/models/resnet18_mask_aware.pth")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Zero fill in normalized space corresponds to -mean / std
    zero_norm = torch.tensor([(-m / s) for m, s in zip(NORM_MEAN, NORM_STD)], device=device).view(1, 3, 1, 1)
    
    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (imgs, labels, _, _) in enumerate(train_loader):
            B = imgs.size(0)
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Apply random masks to 80% of batch
            masks = []
            for _ in range(B):
                if rng.random() < 0.8:
                    b = rng.choice([(0.10, 0.30), (0.30, 0.60), (0.60, 0.80)])
                    m = sim.generate_shape_mask(b, rng=rng)  # held-out mask family (test uses generate_cloud_mask)
                else:
                    m = np.zeros((64, 64), dtype=np.uint8)
                masks.append(m)
                
            masks_t = torch.from_numpy(np.stack(masks, axis=0)[:, np.newaxis, :, :]).float().to(device)
            masked_imgs = imgs * (1.0 - masks_t) + zero_norm * masks_t
            
            optimizer.zero_grad()
            outputs = model(masked_imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * B
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += B
            
        train_acc = correct / total
        
        # Validation on masked validation set
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_imgs, val_labels, _, _ in val_loader:
                B_v = val_imgs.size(0)
                val_imgs, val_labels = val_imgs.to(device), val_labels.to(device)
                
                v_masks = [sim.generate_shape_mask((0.30, 0.60), rng=rng) for _ in range(B_v)]
                v_masks_t = torch.from_numpy(np.stack(v_masks, axis=0)[:, np.newaxis, :, :]).float().to(device)
                v_masked_imgs = val_imgs * (1.0 - v_masks_t) + zero_norm * v_masks_t
                
                v_out = model(v_masked_imgs)
                val_correct += (v_out.argmax(dim=1) == val_labels).sum().item()
                val_total += B_v
                
        val_acc = val_correct / val_total
        elapsed = time.time() - t0
        print(f"Mask-Aware Epoch [{epoch}/{epochs}] ({elapsed:.1f}s) | Train Acc: {train_acc:.4f} | Masked Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved best Mask-Aware Classifier checkpoint (Masked Val Acc: {val_acc:.4f})")
            
    print(f"Mask-Aware Classifier training complete. Checkpoint at {save_path}")
    return model

if __name__ == "__main__":
    train_mask_aware_classifier()
