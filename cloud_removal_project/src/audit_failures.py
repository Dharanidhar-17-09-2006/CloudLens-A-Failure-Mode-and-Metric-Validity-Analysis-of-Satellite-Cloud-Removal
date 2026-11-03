"""
Fill-failure analysis (fills only, standard/clean classifier).

1. Prediction-shift analysis (automatic, data-backed): of the harmful flips caused by FILL
   methods, what share lands in SeaLake/River (flat/dark-tone collapse), per method.
   This is a statement about predicted classes, not a visual failure label.
2. Manual failure labeling: writes results/manual_label_sheet.csv (one row per sampled harmful
   flip, with a montage image path: clean | cloudy | fill | mask). YOU fill the `manual_label`
   column by looking at the montage, using exactly one of:
       color_shift, texture_hallucination, lost_structure, edge_blur, other
   Re-run this script afterwards to get the label distribution. An existing sheet is never overwritten.
Cloudy-as-is and masked inputs involve no fill and are excluded on purpose.
"""
from pathlib import Path
import pandas as pd
import numpy as np

RES = Path("cloud_removal_project/results")
CLASSES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
           'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
FILLS = ["mean_fill", "nearest_neighbor", "telea_inpaint", "unet"]
LABELS = ["color_shift", "texture_hallucination", "lost_structure", "edge_blur", "other"]
N_LABEL = 60


def main():
    df = pd.read_csv(RES / "per_image_metrics.csv")
    fills = df[(df["classifier"] == "clean") & (df["method"].isin(FILLS)) & (df["condition"] == "thick")].copy()
    fills["pred_class"] = fills["method_pred"].map(dict(enumerate(CLASSES)))
    harm = fills[fills["flip_type"] == "harmful_flip"].copy()
    print(f"Harmful flips from fills (thick clouds, clean classifier): {len(harm)}")

    # 1. prediction-shift table
    rows = []
    for (m, b), g in harm.groupby(["method", "coverage_bin"]):
        n_fill_rows = len(fills[(fills["method"] == m) & (fills["coverage_bin"] == b)])
        rows.append({"method": m, "coverage_bin": b, "n_rows": n_fill_rows, "harmful_flips": len(g),
                     "harmful_rate_%": round(100 * len(g) / n_fill_rows, 2),
                     "share_into_SeaLake_or_River_%": round(100 * g["pred_class"].isin(["SeaLake", "River"]).mean(), 1),
                     "top_wrong_class": g["pred_class"].value_counts().index[0]})
    shift = pd.DataFrame(rows)
    shift.to_csv(RES / "prediction_shift_fills.csv", index=False)
    print("\nPrediction-shift analysis (fills only):\n", shift.to_string(index=False))
    per_method = harm.groupby("method")["pred_class"].apply(lambda s: round(100 * s.isin(["SeaLake", "River"]).mean(), 1))
    print("\nShare of harmful flips predicted SeaLake/River, per fill method (%):\n", per_method.to_string())

    # 2. manual labeling sheet (never overwrite)
    sheet = RES / "manual_label_sheet.csv"
    if sheet.exists():
        lab = pd.read_csv(sheet)
        done = lab[lab["manual_label"].isin(LABELS)]
        print(f"\nManual labels found: {len(done)}/{len(lab)}")
        if len(done):
            print(done["manual_label"].value_counts().to_string())
            print("\nBy method:\n", pd.crosstab(done["method"], done["manual_label"]).to_string())
            pd.crosstab(done["method"], done["manual_label"]).to_csv(RES / "manual_label_summary.csv")
        else:
            print(f"Fill the manual_label column in {sheet} with one of {LABELS}, then re-run.")
        return
    harm = harm[harm["audit_image"].notna() & (harm["audit_image"] != "")]
    take = []
    per = max(1, N_LABEL // harm["method"].nunique())
    for _, g in harm.groupby("method"):
        take.append(g.sample(min(len(g), per), random_state=42))
    out = pd.concat(take)[["audit_image", "image_id", "class_name", "pred_class", "method", "coverage_bin",
                           "coverage_pct", "psnr_mask", "ssim_mask", "lpips"]].reset_index(drop=True)
    out["manual_label"] = ""
    out.to_csv(sheet, index=False)
    print(f"\nWrote {len(out)} rows to {sheet}. Montage layout: clean | cloudy | fill | mask. Allowed labels: {LABELS}")


if __name__ == "__main__":
    main()
