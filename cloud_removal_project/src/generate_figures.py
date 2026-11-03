"""
Generate visualization figures for:
1. Downstream Accuracy vs Cloud Coverage (with 95% Confidence Intervals)
2. Harmful Flip Rate (Misleading predictions caused by fabricated fills)
3. Metric Validity Summary (Pooled & Within-Bin)
4. Empirical Failure Taxonomy Distribution (Water Body Shift, Texture Smear, etc.)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
fig_dir = Path("cloud_removal_project/results/figures")
fig_dir.mkdir(parents=True, exist_ok=True)

RES = Path("cloud_removal_project/results")
BINS = ["10-30%", "30-60%", "60-80%"]
METHODS = ["cloudy_as_is", "masked", "mean_fill", "nearest_neighbor", "telea_inpaint", "unet"]
CLFS = ["clean", "mask_aware", "fill_aware"]
CLF_TITLE = {"clean": "Standard classifier (clean-trained)", "mask_aware": "Mask-aware classifier", "fill_aware": "Fill-aware classifier"}
LABELS = {"cloudy_as_is": "Cloudy as-is", "masked": "Masked (zeros)", "mean_fill": "Mean fill",
          "nearest_neighbor": "Nearest neighbor", "telea_inpaint": "Telea", "unet": "U-Net"}
COLORS = {"cloudy_as_is": "#7f7f7f", "masked": "#1f77b4", "mean_fill": "#ff7f0e",
          "nearest_neighbor": "#2ca02c", "telea_inpaint": "#9467bd", "unet": "#d62728"}

acc_csv = RES / "downstream_accuracy_table.csv"
if acc_csv.exists():
    df_acc = pd.read_csv(acc_csv)
    thick = df_acc[df_acc["condition"] == "thick"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, clf in zip(axes, CLFS):
        for m in METHODS:
            sub = thick[(thick["classifier"] == clf) & (thick["method"] == m)].set_index("coverage_bin")
            ax.errorbar(BINS, [sub.loc[b, "accuracy_pct"] for b in BINS], yerr=[sub.loc[b, "ci_95_pct"] for b in BINS],
                        marker="o", capsize=3, label=LABELS[m], color=COLORS[m])
        ax.set_title(CLF_TITLE[clf], fontweight="bold", fontsize=11)
        ax.set_xlabel("Cloud coverage bin")
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("Accuracy (%) with 95% CI")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "downstream_accuracy_by_bin.png", dpi=300)
    plt.close()
    print("Saved figure: downstream_accuracy_by_bin.png")

flip_csv = RES / "flip_analysis.csv"
if flip_csv.exists():
    df_flip = pd.read_csv(flip_csv)
    thick_flip = df_flip[df_flip["condition"] == "thick"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    x = np.arange(len(BINS)); width = 0.13
    for ax, clf in zip(axes, CLFS):
        for i, m in enumerate(METHODS):
            sub = thick_flip[(thick_flip["classifier"] == clf) & (thick_flip["method"] == m)].set_index("coverage_bin")
            ax.bar(x + (i - len(METHODS) / 2 + 0.5) * width, [sub.loc[b, "harmful_rate_%"] for b in BINS], width,
                   yerr=[sub.loc[b, "harmful_ci_95_%"] for b in BINS], capsize=2, label=LABELS[m], color=COLORS[m])
        ax.set_title(CLF_TITLE[clf], fontweight="bold", fontsize=11)
        ax.set_xticks(x); ax.set_xticklabels(BINS)
        ax.set_xlabel("Cloud coverage bin")
    axes[0].set_ylabel("Harmful flip rate (%)")
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "harmful_flip_rate.png", dpi=300)
    plt.close()
    print("Saved figure: harmful_flip_rate.png")

per_img_csv = RES / "per_image_metrics.csv"
if per_img_csv.exists():
    df_pi = pd.read_csv(per_img_csv)
    df_pi = df_pi[(df_pi["classifier"] == "clean") & (df_pi["method"].isin(["mean_fill", "nearest_neighbor", "telea_inpaint", "unet"]))]
    harmful_df = df_pi[df_pi["flip_type"] == "harmful_flip"]
    correct_df = df_pi[df_pi["flip_type"] == "retained_correct"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, col, title in zip(axes, ["psnr_full", "ssim_full", "lpips"], ["Full PSNR (dB)", "Full SSIM", "LPIPS (lower = better)"]):
        ax.boxplot([correct_df[col].dropna(), harmful_df[col].dropna()], tick_labels=["Retained correct", "Harmful flip"])
        ax.set_title(title + " - fills only", fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "metric_validity_boxplots.png", dpi=300)
    plt.close()
    print("Saved figure: metric_validity_boxplots.png")

print("All figures regenerated successfully.")
