"""
Phase 4: ResNet-18 Downstream Land-Cover Classifier.
Trains on clean EuroSAT images to establish a strong classification baseline.
Evaluates classification performance across methods and coverage bins.
"""

import os
from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import numpy as np

from dataset import EuroSATDataset

# ImageNet normalization
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

def get_classifier(num_classes=10, pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_classifier(epochs=3, batch_size=64, lr=1e-3, device="cpu"):
    print(f"--- Training ResNet-18 Classifier on Clean EuroSAT (device: {device}) ---")
    
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    
    train_ds = EuroSATDataset("cloud_removal_project/data/splits/train.csv", transform=train_transform)
    val_ds = EuroSATDataset("cloud_removal_project/data/splits/val.csv", transform=val_transform)
    
    # Use subset for fast CPU convergence if needed, or full
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = get_classifier(num_classes=10, pretrained=True).to(device)
    
    # Fine-tune layer3, layer4 and fc for fast high accuracy
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True
        
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    best_acc = 0.0
    model_save_path = Path("cloud_removal_project/models/resnet18_clean.pth")
    model_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (imgs, labels, _, _) in enumerate(train_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
                
        train_acc = correct / total
        train_loss = running_loss / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels, _, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                
        val_acc = val_correct / val_total
        elapsed = time.time() - t0
        print(f"Epoch {epoch} finished in {elapsed:.1f}s | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"--> Saved best model with Val Acc: {val_acc:.4f}")
            
    print(f"Classifier training completed. Best Val Acc: {best_acc:.4f}")
    return model

def load_trained_classifier(checkpoint_path="cloud_removal_project/models/resnet18_clean.pth", device="cpu"):
    model = get_classifier(num_classes=10, pretrained=False)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded classifier from {checkpoint_path}")
    else:
        print(f"Checkpoint not found at {checkpoint_path}")
    model.to(device)
    model.eval()
    return model

if __name__ == "__main__":
    train_classifier(epochs=2, batch_size=128)
