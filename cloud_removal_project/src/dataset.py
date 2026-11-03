"""
Dataset loader for EuroSAT splits.
"""

from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class EuroSATDataset(Dataset):
    def __init__(self, split_csv, base_dir="cloud_removal_project", transform=None):
        self.base_dir = Path(base_dir)
        self.df = pd.read_csv(split_csv)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.base_dir / row["rel_path"]
        with Image.open(img_path) as img:
            image = img.convert("RGB")
            
        label = int(row["label"])
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, label, row["filename"], row["class_name"]
