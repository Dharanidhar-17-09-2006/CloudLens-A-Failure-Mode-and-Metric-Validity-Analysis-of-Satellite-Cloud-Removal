"""
RICE data for training/evaluating inpainters (numpy only, no torch).

All images are resized 512 -> 256, so a 64x64 crop equals a 128x128 RICE patch at half scale
(the same patches used in the earlier real-cloud evaluation).
Splits are by IMAGE (70/10/20) and saved to results/rice_inpaint/splits.json so training and
evaluation always use the same held-out images.
Training uses only: clear images of the TRAIN split (RICE1 + RICE2) and real cloud masks of the
RICE2 TRAIN split. Evaluation uses the real cloudy tile + real mask of the TEST split.
"""
import json
import numpy as np, cv2
import real_cloud_eval as R

DATA = R.ROOT / "data" / "RICE_DATASET"
OUT = R.ROOT / "results" / "rice_inpaint"
SIZE = 256


def load_set(name, thr=25):
    base = DATA / name
    clouds, clears = R.list_images(base / "cloud"), R.list_images(base / "label")
    masks = R.list_images(base / "mask") if (base / "mask").exists() else {}
    out = {}
    for k in sorted(set(clouds) & set(clears)):
        cl, cr = R.read_rgb(clouds[k]), R.read_rgb(clears[k])
        if cr.shape != cl.shape:
            cr = cv2.resize(cr, (cl.shape[1], cl.shape[0]), interpolation=cv2.INTER_AREA)
        if k in masks:
            m = cv2.imread(str(masks[k]), cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (cl.shape[1], cl.shape[0]), interpolation=cv2.INTER_NEAREST)
            m = (m > (127 if m.max() > 1 else 0)).astype(np.uint8)
        else:
            m, _ = R.estimate_mask(cl, cr, thr)      # RICE1: estimated from |cloudy - clear|
        rs = lambda a: cv2.resize(a, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        out[k] = dict(cloud=rs(cl), clear=rs(cr),
                      mask=(cv2.resize(m.astype(np.float32), (SIZE, SIZE), interpolation=cv2.INTER_AREA) > 0.5).astype(np.uint8))
    return out


def make_splits(keys, seed=0, frac=(0.7, 0.1)):
    keys = sorted(keys)
    perm = np.random.default_rng(seed).permutation(len(keys))
    a = int(frac[0] * len(keys)); b = a + int(frac[1] * len(keys))
    return {"train": [keys[i] for i in perm[:a]], "val": [keys[i] for i in perm[a:b]], "test": [keys[i] for i in perm[b:]]}


def get_data():
    """returns rice1, rice2 (dict stem -> arrays) and splits {'RICE1': {...}, 'RICE2': {...}}"""
    OUT.mkdir(parents=True, exist_ok=True)
    rice1, rice2 = load_set("RICE1"), load_set("RICE2")
    sp = OUT / "splits.json"
    if sp.exists():
        splits = json.load(open(sp))
    else:
        splits = {"RICE1": make_splits(rice1), "RICE2": make_splits(rice2)}
        json.dump(splits, open(sp, "w"))
    print({k: {s: len(v) for s, v in d.items()} for k, d in splits.items()})
    return rice1, rice2, splits


def eval_tiles(items, keys, tile=64, lo=0.10, hi=0.80):
    out = []
    for k in keys:
        d = items[k]
        for y in range(0, SIZE - tile + 1, tile):
            for x in range(0, SIZE - tile + 1, tile):
                m = d["mask"][y:y + tile, x:x + tile]
                cov = float(m.mean())
                if lo <= cov <= hi:
                    out.append(dict(image_id=k, cloudy=d["cloud"][y:y + tile, x:x + tile],
                                    clear=d["clear"][y:y + tile, x:x + tile], mask=m, coverage=cov, bin=R.bin_of(cov)))
    return out


class TrainSampler:
    """Random 64x64 clear crops + random real-mask crops (10-80% coverage), with flips/rot90."""
    def __init__(self, rice1, rice2, splits, tile=64, seed=0):
        self.tile = tile
        self.rng = np.random.default_rng(seed)
        self.clears = [rice1[k]["clear"] for k in splits["RICE1"]["train"]] + [rice2[k]["clear"] for k in splits["RICE2"]["train"]]
        self.masks = [rice2[k]["mask"] for k in splits["RICE2"]["train"]]
        assert self.clears and self.masks, "no training images/masks found"

    def _aug(self, a):
        if self.rng.random() < 0.5:
            a = a[:, ::-1]
        return np.ascontiguousarray(np.rot90(a, int(self.rng.integers(0, 4))))

    def _crop(self, arr):
        S, t = arr.shape[0], self.tile
        y, x = self.rng.integers(0, S - t + 1, 2)
        return arr[y:y + t, x:x + t]

    def one(self):
        img = self._aug(self._crop(self.clears[self.rng.integers(len(self.clears))]))
        img = np.clip(img.astype(np.float32) * self.rng.uniform(0.9, 1.1), 0, 255).astype(np.uint8)
        for _ in range(50):
            m = self._aug(self._crop(self.masks[self.rng.integers(len(self.masks))]))
            if 0.10 <= m.mean() <= 0.80:
                break
        return img, m

    def batch(self, B):
        pairs = [self.one() for _ in range(B)]
        return np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])
