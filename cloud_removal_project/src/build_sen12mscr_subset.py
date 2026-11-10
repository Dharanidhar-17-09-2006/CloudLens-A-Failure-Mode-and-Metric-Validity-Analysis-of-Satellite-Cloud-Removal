"""
Build the labelled real-cloud subset:
SEN12MS-CR cloudy / cloud-free RGB pairs + SEN12MS IGBP land cover.

Three steps, run in this order:
  --inspect : list land-cover archive members.
  --stream  : stream pairs, estimate cloud masks, save patches.
              Supports automatic checkpointing and resuming.
  --label   : join IGBP labels, cut 64x64 tiles, keep pure tiles,
              split by scene.

Choices:
  * RGB = B4, B3, B2, reflectance / 3000 clipped to [0,1].
  * Cloud mask = cloudy RGB mean brighter than cloud-free RGB mean
    by more than --thr. A 3x3 morphological opening is applied.
    This is an ESTIMATE, not a ground-truth cloud mask.
  * Tiles: 4 non-overlapping 64x64 crops per 256x256 patch.
  * Label: dominant simplified-IGBP class, provided it covers
    >= --purity of valid pixels and >= 90% of pixels are valid.
  * Split: by (season, scene) hash, 70/10/20.

Run:
  python cloud_removal_project/src/build_sen12mscr_subset.py --stream --per_season 250
  python cloud_removal_project/src/build_sen12mscr_subset.py --label

Streaming automatically resumes from an existing checkpoint.
"""

import argparse
import glob
import hashlib
import io
import os
import tarfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


RGB = [3, 2, 1]  # B4, B3, B2
SEASONS = ["spring", "summer", "fall", "winter"]

ROI = {
    "spring": "ROIs1158",
    "summer": "ROIs1868",
    "fall": "ROIs1970",
    "winter": "ROIs2017",
}

IGBP_TO_10 = {
    1: 1, 2: 1, 3: 1, 4: 1, 5: 1,
    6: 2, 7: 2,
    8: 3, 9: 3,
    10: 4,
    11: 5,
    12: 6,
    13: 7,
    14: 6,
    15: 8,
    16: 9,
    17: 10,
}

CLASS_NAMES = [
    "Forest", "Shrubland", "Savanna", "Grassland", "Wetlands",
    "Croplands", "Urban", "Snow/Ice", "Barren", "Water",
]

SCALE = 0.3  # reflectance value mapped to 255
CROPS = [(32, 32), (32, 160), (160, 32), (160, 160)]
TILE = 64
CHECKPOINT_INTERVAL = 50


def inspect(lc_dir):
    for archive in sorted(glob.glob(str(Path(lc_dir) / "*.tar.gz"))):
        print(Path(archive).name)


def to_u8(x):
    return np.clip(x / SCALE * 255.0, 0, 255).round().astype(np.uint8)


def process_pair(c13, t13, thr):
    """Convert cloudy and clear images to RGB and estimate a cloud mask."""
    cloudy = c13[..., RGB]
    target = t13[..., RGB]

    mask = ((cloudy.mean(-1) - target.mean(-1)) > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    return to_u8(cloudy), to_u8(target), mask


def save_checkpoint(C, T, M, keys, out_path):
    """Save collected pairs and their metadata."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    keys_path = out_path.with_suffix(".keys.csv")
    temp_path = out_path.with_name(out_path.stem + ".tmp.npz")

    key_array = np.asarray(keys, dtype=str)

    with open(temp_path, "wb") as f:
        np.savez_compressed(
            f,
            cloudy=np.stack(C),
            target=np.stack(T),
            mask=np.stack(M),
            season=key_array[:, 0],
            scene=key_array[:, 1],
            patch=key_array[:, 2],
        )

    # Replace the previous checkpoint only after the new one is saved.
    os.replace(temp_path, out_path)

    pd.DataFrame(keys, columns=["season", "scene", "patch"]).to_csv(keys_path, index=False)

    print(f"Checkpoint saved: {len(keys)} pairs")


def load_checkpoint(out_path):
    """Load previous progress if a checkpoint exists."""
    C, T, M, keys = [], [], [], []

    out_path = Path(out_path)
    keys_path = out_path.with_suffix(".keys.csv")

    if not out_path.exists():
        return C, T, M, keys

    print(f"Loading checkpoint from {out_path}")

    with np.load(out_path) as old:
        C = list(old["cloudy"])
        T = list(old["target"])
        M = list(old["mask"])

        if all(k in old.files for k in ["season", "scene", "patch"]):
            keys = list(zip(
                old["season"].astype(str),
                old["scene"].astype(str),
                old["patch"].astype(str),
            ))

    # Support checkpoints created by the original script.
    if not keys and keys_path.exists():
        old_keys = pd.read_csv(keys_path, dtype=str)
        keys = list(old_keys.itertuples(index=False, name=None))

    if not (len(C) == len(T) == len(M) == len(keys)):
        raise ValueError("Checkpoint arrays and metadata have different lengths.")

    return C, T, M, keys


def stream(per_season, max_stream, thr, out_path, max_per_scene=20, buffer=300):
    from datasets import load_dataset

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Restore previously collected samples.
    C, T, M, keys = load_checkpoint(out_path)

    quota = {season: per_season for season in SEASONS}

    for season, scene, patch in keys:
        if season in quota:
            quota[season] -= 1

    if any(v < 0 for v in quota.values()):
        raise ValueError("Existing checkpoint exceeds the requested quota.")

    seen = set(keys)

    # Count pairs per (season, scene) so no scene dominates the subset.
    scene_count = {}
    for s_, sc, _ in keys:
        scene_count[(s_, sc)] = scene_count.get((s_, sc), 0) + 1

    print(f"Already collected: {len(keys)} pairs")
    print(f"Remaining quota: {quota}")

    if all(v <= 0 for v in quota.values()):
        print("All season quotas are already completed.")
        return

    print("Connecting to SEN12MS-CR...")

    ds = load_dataset(
        "Hermanni/sen12mscr", split="train", streaming=True
    ).shuffle(seed=0, buffer_size=buffer)

    new_count = 0
    n = 0

    try:
        for n, sample in enumerate(ds, 1):
            if n > max_stream or all(v <= 0 for v in quota.values()):
                break

            season = sample["season"]

            if quota.get(season, 0) <= 0:
                continue

            key = (season, str(sample["scene"]), str(sample["patch"]))

            if scene_count.get((season, key[1]), 0) >= max_per_scene:
                continue

            if key in seen:
                continue

            shape = tuple(sample["opt_shape"])

            cloudy = np.frombuffer(
                sample["cloudy"], dtype=np.int16
            ).reshape(shape).astype(np.float32) / 10000.0

            target = np.frombuffer(
                sample["target"], dtype=np.int16
            ).reshape(shape).astype(np.float32) / 10000.0

            cloudy_u8, target_u8, mask = process_pair(cloudy, target, thr)

            C.append(cloudy_u8)
            T.append(target_u8)
            M.append(mask)
            keys.append(key)
            seen.add(key)
            scene_count[(season, key[1])] = scene_count.get((season, key[1]), 0) + 1

            quota[season] -= 1
            new_count += 1

            # Save progress every CHECKPOINT_INTERVAL newly collected pairs.
            if new_count % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(C, T, M, keys, out_path)

            if new_count % 100 == 0:
                print(f"Kept {len(keys)} pairs (rows read {n}) | quota left {quota}")

    except KeyboardInterrupt:
        print("\nStreaming interrupted by user.")

    except Exception as e:
        print(f"\nStreaming error: {e}")
        raise

    finally:
        # Save any progress collected after the last checkpoint.
        if new_count > 0:
            print("Saving latest progress...")
            save_checkpoint(C, T, M, keys, out_path)

    print("\nStreaming summary")
    print("-----------------")
    print(f"Rows read: {n}")
    print(f"New pairs collected: {new_count}")
    print(f"Total pairs collected: {len(keys)}")
    print(f"Remaining quota: {quota}")

    if any(v > 0 for v in quota.values()):
        print("Some season quotas remain. Run the same command again to resume.")
    else:
        print("All season quotas completed.")


def igbp_band(raw):
    import tifffile

    arr = tifffile.imread(io.BytesIO(raw))

    if arr.ndim == 2:
        return arr
    if arr.shape[0] == 4:
        return arr[0]
    if arr.shape[-1] == 4:
        return arr[..., 0]

    raise ValueError(f"Unexpected land-cover TIFF shape {arr.shape}")


def split_of(season, scene):
    h = int(hashlib.md5(f"42_{season}_{scene}".encode()).hexdigest(), 16) % 100

    if h < 70:
        return "train"
    if h < 80:
        return "val"
    return "test"


def label(lc_dir, pairs_path, out_path, purity):
    # NpzFile decompresses an array on EVERY access, so each array is loaded once here.
    with np.load(pairs_path) as d:
        cloudy_all = d["cloudy"]
        target_all = d["target"]
        mask_all = d["mask"]

    keys = pd.read_csv(Path(pairs_path).with_suffix(".keys.csv"), dtype=str)

    need = {}
    remaining = {season: 0 for season in SEASONS}

    for i, (season, scene, patch) in enumerate(keys.itertuples(index=False)):
        name = (
            f"{ROI[season]}_{season}/"
            f"lc_{scene}/"
            f"{ROI[season]}_{season}_lc_{scene}_{patch}.tif"
        )
        need[name] = i
        remaining[season] += 1

    lc = {}

    for archive in sorted(glob.glob(str(Path(lc_dir) / "*.tar.gz"))):
        archive_name = Path(archive).name
        season = next(s for s in SEASONS if s in archive_name)

        print(f"Scanning {archive_name} (need {remaining[season]})...")

        with tarfile.open(archive, "r|gz") as tar:
            for member in tar:
                if member.name in need:
                    f = tar.extractfile(member)
                    if f is None:
                        continue

                    lc[need[member.name]] = igbp_band(f.read())
                    remaining[season] -= 1

                    if remaining[season] == 0:
                        break

    print(f"Matched {len(lc)} of {len(keys)} patches")

    if not lc:
        print("No matches found. Expected names look like:")
        print("\n".join(list(need)[:5]))
        return

    vals, counts = np.unique(
        np.concatenate([b.ravel() for b in lc.values()]),
        return_counts=True,
    )
    print("Raw IGBP values found:", dict(zip(vals.tolist(), counts.tolist())))

    lut = np.zeros(256, np.int16)
    for raw_class, simplified_class in IGBP_TO_10.items():
        lut[raw_class] = simplified_class

    rows = {k: [] for k in [
        "cloudy", "target", "mask", "label", "coverage",
        "split", "season", "scene", "patch",
    ]}

    for i, band in lc.items():
        season, scene, patch = keys.iloc[i]
        m10 = lut[band]

        for r, c in CROPS:
            crop = m10[r:r + TILE, c:c + TILE]
            valid = crop > 0

            if valid.mean() < 0.9:
                continue

            counts = np.bincount(crop[valid], minlength=11)
            dominant = int(counts.argmax())

            if counts[dominant] / valid.sum() < purity:
                continue

            mask = mask_all[i, r:r + TILE, c:c + TILE]

            rows["cloudy"].append(cloudy_all[i, r:r + TILE, c:c + TILE])
            rows["target"].append(target_all[i, r:r + TILE, c:c + TILE])
            rows["mask"].append(mask)
            rows["label"].append(dominant - 1)
            rows["coverage"].append(float(mask.mean()))
            rows["split"].append(split_of(season, scene))
            rows["season"].append(season)
            rows["scene"].append(scene)
            rows["patch"].append(patch)

    out = {k: np.array(v) for k, v in rows.items()}

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)

    df = pd.DataFrame({
        "split": out["split"],
        "label": out["label"],
        "coverage": out["coverage"],
    })

    print(f"\nSaved {len(df)} tiles to {out_path}")
    print("Tiles per split:", df.split.value_counts().to_dict())

    if len(df) > 0:
        print("\nClass counts per split:")
        print(pd.crosstab(df.label.map(lambda i: CLASS_NAMES[i]), df.split).to_string())

        bins = pd.cut(
            df[df.split == "test"].coverage,
            [0, 0.10, 0.30, 0.60, 0.80, 1.0],
            include_lowest=True,
        )
        print("\nTest tiles by estimated coverage:",
              bins.value_counts().sort_index().to_dict())
    else:
        print("No tiles passed the purity and validity filters.")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--label", action="store_true")

    ap.add_argument("--lc_dir", default="cloud_removal_project/data/SEN12MS_lc")
    ap.add_argument("--per_season", type=int, default=250)
    ap.add_argument("--max_stream", type=int, default=100000)
    ap.add_argument("--thr", type=float, default=0.05)
    ap.add_argument("--purity", type=float, default=0.7)
    ap.add_argument("--max_per_scene", type=int, default=20)
    ap.add_argument("--buffer", type=int, default=300)

    ap.add_argument("--pairs", default="cloud_removal_project/data/sen12mscr_pairs.npz")
    ap.add_argument("--tiles", default="cloud_removal_project/data/sen12mscr_tiles.npz")

    args = ap.parse_args()

    if args.inspect:
        inspect(args.lc_dir)

    if args.stream:
        stream(args.per_season, args.max_stream, args.thr, args.pairs,
               args.max_per_scene, args.buffer)

    if args.label:
        label(args.lc_dir, args.pairs, args.tiles, args.purity)

    if not (args.inspect or args.stream or args.label):
        ap.print_help()


if __name__ == "__main__":
    main()
