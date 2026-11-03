import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


class PairDataset(Dataset):
    def __init__(self, root, ids_file, cloudy_dir="cloud", clear_dir="label", mask_dir=None, ext=".png", mask_thr=0.1):
        self.root, self.cloudy_dir, self.clear_dir = root, cloudy_dir, clear_dir
        self.mask_dir, self.ext, self.mask_thr = mask_dir, ext, mask_thr
        with open(ids_file) as f:
            self.ids = [l.strip() for l in f if l.strip()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        id_ = self.ids[i]
        cloudy = load_rgb(os.path.join(self.root, self.cloudy_dir, id_ + self.ext))
        clear = load_rgb(os.path.join(self.root, self.clear_dir, id_ + self.ext))
        if self.mask_dir:
            m = np.asarray(Image.open(os.path.join(self.root, self.mask_dir, id_ + self.ext)).convert("L"), dtype=np.float32) / 255.0
            mask = (m > 0.5).astype(np.float32)
        else:
            mask = (np.abs(cloudy - clear).mean(-1) > self.mask_thr).astype(np.float32)
        t = lambda x: torch.from_numpy(x.transpose(2, 0, 1).copy())
        return {"id": id_, "cloudy": t(cloudy), "clear": t(clear), "mask": torch.from_numpy(mask)[None]}
