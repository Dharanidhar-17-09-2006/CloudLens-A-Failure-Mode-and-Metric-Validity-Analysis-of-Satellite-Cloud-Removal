"""
Classifier-free + stability-filtered evaluation on real clouds (RICE).

The EuroSAT classifiers are unstable on RICE tiles (diagnostics: clear vs lightly degraded clear
agrees only 31-41%), so plain agreement is not informative. This script adds two fairer views:
  1. FIDELITY: PSNR / SSIM (inside the cloud mask) and LPIPS of each fill vs the paired CLEAR tile.
     No classifier involved. Same metrics on synthetic clouds for comparison.
  2. STABLE-TILE AGREEMENT: agreement with the clear-tile prediction, using only tiles where the
     classifier's clear-tile prediction survives a light degradation (so the reference is meaningful).

  python cloud_removal_project/src/real_fidelity_eval.py --cloud_dir ... --clear_dir ... [--mask_dir ...] --tag rice2
"""
import argparse
import numpy as np, pandas as pd
import real_cloud_eval as R
from real_diagnostics import probs


def run(items, source, models, unet, device, evaluator):
    outs = {m: [] for m in R.METHODS}
    for it in items:
        o = R.fill_all(unet, device, it["cloudy"], it["mask"])
        for m in R.METHODS:
            outs[m].append(o[m])

    frows = []
    for i, it in enumerate(items):
        for m in R.METHODS:
            met = evaluator.evaluate_all(it["clear"], outs[m][i], it["mask"])
            frows.append(dict(source=source, tile=i, image_id=it["image_id"], bin=it["bin"], method=m, **met))

    rng = np.random.default_rng(0)
    clears = [it["clear"] for it in items]
    degraded = [np.clip(c.astype(np.float32) * rng.uniform(0.9, 1.1) + rng.normal(0, 5, c.shape), 0, 255).astype(np.uint8) for c in clears]
    arows = []
    for cn, model in models.items():
        pc = probs(model, clears, device).argmax(1)
        stable = pc == probs(model, degraded, device).argmax(1)
        for m in R.METHODS:
            pm = probs(model, outs[m], device).argmax(1)
            for i, it in enumerate(items):
                arows.append(dict(source=source, tile=i, image_id=it["image_id"], bin=it["bin"], classifier=cn,
                                  method=m, agree=int(pm[i] == pc[i]), stable=int(stable[i]), clear_pred=int(pc[i])))
    return pd.DataFrame(frows), pd.DataFrame(arows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud_dir", required=True); ap.add_argument("--clear_dir", required=True)
    ap.add_argument("--mask_dir", default=None); ap.add_argument("--tag", default="rice")
    ap.add_argument("--tile", type=int, default=128); ap.add_argument("--thr", type=float, default=25)
    ap.add_argument("--max_tiles", type=int, default=1500); ap.add_argument("--n_synth", type=int, default=300)
    a = ap.parse_args()
    import torch
    from metrics import MetricEvaluator
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = R.ROOT / "results" / "real_clouds" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    models, unet = R.load_models(device)
    evaluator = MetricEvaluator(use_lpips=True, device=device)

    real = R.real_items(a.cloud_dir, a.clear_dir, a.mask_dir, a.tile, a.thr, a.max_tiles)
    f_r, a_r = run(real, "real", models, unet, device, evaluator)
    f_s, a_s = run(R.synthetic_items(a.n_synth), "synthetic", models, unet, device, evaluator)
    fid, agr = pd.concat([f_r, f_s]), pd.concat([a_r, a_s])
    fid.to_csv(out / "fidelity_per_tile.csv", index=False)

    summ = fid.groupby(["source", "bin", "method"])[["psnr_mask", "ssim_mask", "lpips"]].mean().round(3).reset_index()
    summ.to_csv(out / "fidelity_summary.csv", index=False)

    def stable_table(df):
        rows = []
        for (s, c, b, m), g in df.groupby(["source", "classifier", "bin", "method"]):
            gs = g[g["stable"] == 1]
            if len(gs) < 20:
                continue
            mean, lo, hi = R.cluster_ci(gs)
            rows.append(dict(source=s, classifier=c, bin=b, method=m, n_stable=len(gs), frac_stable=round(g["stable"].mean(), 3),
                             agreement_pct=100 * mean, ci_low=100 * lo, ci_high=100 * hi))
        return pd.DataFrame(rows)

    st = stable_table(agr); st.to_csv(out / "stable_agreement.csv", index=False)
    # same, but dropping tiles whose clear prediction is River/SeaLake (classes 8, 9): guards against
    # "flat tile -> SeaLake" agreement that does not reflect a good fill
    st_ns = stable_table(agr[~agr["clear_pred"].isin([8, 9])]); st_ns.to_csv(out / "stable_agreement_no_water.csv", index=False)

    pd.set_option("display.width", 220)
    for metric in ["lpips", "psnr_mask", "ssim_mask"]:
        print(f"\n=== fidelity: {metric} (fill vs clear tile, inside cloud mask; LPIPS lower = better) ===")
        print(summ.pivot_table(index="method", columns=["source", "bin"], values=metric).round(3).to_string())
    for cn in R.CLFS:
        print(f"\n=== stable-tile agreement % | classifier = {cn} ===")
        sub = st[st.classifier == cn]
        if len(sub):
            print(sub.pivot_table(index="method", columns=["source", "bin"], values="agreement_pct").round(1).to_string())
            print("fraction of tiles stable:\n", sub.groupby(["source", "bin"])["frac_stable"].first().to_string())
    for cn in R.CLFS:
        print(f"\n=== stable-tile agreement %, EXCLUDING tiles whose clear prediction is River/SeaLake | classifier = {cn} ===")
        sub = st_ns[st_ns.classifier == cn]
        if len(sub):
            print(sub.pivot_table(index="method", columns=["source", "bin"], values="agreement_pct").round(1).to_string())
            print("tiles kept (real):\n", sub[sub.source == "real"].groupby("bin")["n_stable"].first().to_string())
    print(f"\nSaved to {out}")

if __name__ == "__main__":
    main()
