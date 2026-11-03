"""
Diagnostics for the real-cloud (RICE) results: is the classifier even meaningful on RICE tiles,
and what is the agreement ceiling?  Run BEFORE building bigger models.

For each RICE set and classifier it prints:
  - top predicted classes on CLEAR tiles + mean confidence (collapse / domain-shift check)
  - agreement(clear tile, slightly degraded clear tile)   -> classifier stability ceiling
  - agreement(near-clear cloudy tile (<5% cloud), clear tile) -> ceiling from pair mismatch
    (different date / sensor / resolution between the cloudy and clear image)
Compare these ceilings with the fill agreements: a fill cannot score above them.

  python cloud_removal_project/src/real_diagnostics.py --cloud_dir ... --clear_dir ... [--mask_dir ...] --tag rice2
"""
import argparse
import numpy as np, pandas as pd, cv2
import real_cloud_eval as R

CLASSES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']


def near_clear_tiles(cloud_dir, clear_dir, mask_dir, tile, thr, max_cov, max_tiles, seed=0):
    clouds, clears = R.list_images(cloud_dir), R.list_images(clear_dir)
    masks = R.list_images(mask_dir) if mask_dir else {}
    out = []
    for k in sorted(set(clouds) & set(clears)):
        cl, cr = R.read_rgb(clouds[k]), R.read_rgb(clears[k])
        if cr.shape != cl.shape:
            cr = cv2.resize(cr, (cl.shape[1], cl.shape[0]), interpolation=cv2.INTER_AREA)
        if k in masks:
            m = cv2.imread(str(masks[k]), cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (cl.shape[1], cl.shape[0]), interpolation=cv2.INTER_NEAREST)
            m = (m > (127 if m.max() > 1 else 0)).astype(np.uint8)
        else:
            m, _ = R.estimate_mask(cl, cr, thr)
        H, W = m.shape
        for y in range(0, H - tile + 1, tile):
            for x in range(0, W - tile + 1, tile):
                if m[y:y + tile, x:x + tile].mean() < max_cov:
                    rs = lambda a: cv2.resize(a[y:y + tile, x:x + tile], (64, 64), interpolation=cv2.INTER_AREA)
                    out.append((rs(cl), rs(cr)))
    rng = np.random.default_rng(seed)
    if len(out) > max_tiles:
        out = [out[i] for i in rng.choice(len(out), max_tiles, replace=False)]
    return out


def probs(model, imgs, device, bs=256):
    import torch
    from classifier import NORM_MEAN, NORM_STD
    mean = torch.tensor(NORM_MEAN).view(1, 3, 1, 1); std = torch.tensor(NORM_STD).view(1, 3, 1, 1)
    res = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            x = torch.from_numpy(np.stack(imgs[i:i + bs])).permute(0, 3, 1, 2).float().div(255)
            res.append(torch.softmax(model(((x - mean) / std).to(device)), 1).cpu().numpy())
    return np.concatenate(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud_dir", required=True); ap.add_argument("--clear_dir", required=True)
    ap.add_argument("--mask_dir", default=None); ap.add_argument("--tag", default="rice")
    ap.add_argument("--tile", type=int, default=128); ap.add_argument("--thr", type=float, default=25)
    ap.add_argument("--max_tiles", type=int, default=2000)
    a = ap.parse_args()
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models, _ = R.load_models(device)
    pairs = near_clear_tiles(a.cloud_dir, a.clear_dir, a.mask_dir, a.tile, a.thr, 0.05, a.max_tiles)
    print(f"{len(pairs)} near-clear tiles (<5% cloud)")
    if not pairs:
        raise SystemExit("no near-clear tiles; raise max coverage or check folders")
    cloudy = [p[0] for p in pairs]; clear = [p[1] for p in pairs]
    rng = np.random.default_rng(0)
    degraded = [np.clip(c.astype(np.float32) * rng.uniform(0.9, 1.1) + rng.normal(0, 5, c.shape), 0, 255).astype(np.uint8) for c in clear]
    rows = []
    for cn, model in models.items():
        pc, pd_, pl = probs(model, clear, device), probs(model, degraded, device), probs(model, cloudy, device)
        top = pd.Series(pc.argmax(1)).map(dict(enumerate(CLASSES))).value_counts(normalize=True).head(3)
        rows.append(dict(classifier=cn,
                         top_classes_on_clear=", ".join(f"{k} {100 * v:.0f}%" for k, v in top.items()),
                         mean_confidence_clear=round(float(pc.max(1).mean()), 3),
                         agree_clear_vs_degraded_clear=round(100 * float((pc.argmax(1) == pd_.argmax(1)).mean()), 1),
                         agree_nearclear_cloudy_vs_clear=round(100 * float((pl.argmax(1) == pc.argmax(1)).mean()), 1)))
    df = pd.DataFrame(rows)
    out = R.ROOT / "results" / "real_clouds" / a.tag; out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "diagnostics.csv", index=False)
    pd.set_option("display.width", 220); pd.set_option("display.max_colwidth", 80)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
