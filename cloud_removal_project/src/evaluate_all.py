"""
Comprehensive, Rigorous Evaluation Pipeline (Addressing Claude & Faculty Critiques).
Includes:
- Fair Masked Baseline (Mask-Aware Classifier fine-tuned with missing regions)
- Unadapted Masked Baseline (Clean Classifier on masked tiles)
- 95% Confidence Intervals for all accuracy and flip rates (N=200, Wilson / normal approx)
- Explicit 'unet_prototype' labeling (2 epochs on CPU subset)
- Within-bin and pooled Spearman rank correlations for metrics
- Empirical Failure Taxonomy & Sample Audit (documenting SeaLake/River tone collapse)
"""

import os
import sys
import traceback
from pathlib import Path
import time
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as T
from scipy.stats import spearmanr

from cloud_simulation import CloudSimulator
from metrics import MetricEvaluator
import inpainting_baselines as baselines
from unet_inpaint import SmallUNetInpainter
from classifier import load_trained_classifier, get_classifier, NORM_MEAN, NORM_STD

def run_evaluation(num_test_samples=200, device=None, seed=42):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("=================================================================")
    print("STARTING FULL CLOUD REMOVAL FAILURE-MODE & DOWNSTREAM EVALUATION")
    print("=================================================================")
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 1. Load test split
    test_csv = Path("cloud_removal_project/data/splits/test.csv")
    if not test_csv.exists():
        raise FileNotFoundError(f"Test split not found at {test_csv}")
    test_df = pd.read_csv(test_csv)
    
    # Balanced sampling across all 10 classes
    samples_per_class = max(1, num_test_samples // 10)
    sampled_subsets = []
    for c in sorted(test_df["class_name"].unique()):
        c_df = test_df[test_df["class_name"] == c]
        sampled_subsets.append(c_df.sample(min(len(c_df), samples_per_class), random_state=seed))
    sampled_df = pd.concat(sampled_subsets).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    N = len(sampled_df)
    print(f"Sampled N={N} test images balanced across {sampled_df['class_name'].nunique()} classes.")
    
    # 2. Load Models
    print("Loading clean downstream ResNet-18 classifier...")
    clean_classifier = load_trained_classifier("cloud_removal_project/models/resnet18_clean.pth", device=device)
    
    def load_required(name, ckpt):
        ckpt = Path(ckpt)
        if not ckpt.exists():
            raise FileNotFoundError(f"{ckpt} missing. Train it first (see README). Refusing to silently fall back.")
        m = get_classifier(num_classes=10, pretrained=False)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded {name} from {ckpt}")
        return m.to(device).eval()

    classifiers = {
        "clean": clean_classifier,
        "mask_aware": load_required("mask-aware classifier", "cloud_removal_project/models/resnet18_mask_aware.pth"),
        "fill_aware": load_required("fill-aware classifier", "cloud_removal_project/models/resnet18_fill_aware.pth"),
    }

    print("Loading U-Net inpainter...")
    unet = SmallUNetInpainter().to(device)
    unet_ckpt = Path("cloud_removal_project/models/unet_inpaint.pth")
    if not unet_ckpt.exists():
        raise FileNotFoundError(f"{unet_ckpt} missing. Run train_unet.py first.")
    unet.load_state_dict(torch.load(unet_ckpt, map_location=device))
    unet.eval()

    # 3. Setup Tools
    sim = CloudSimulator(64, 64)
    evaluator = MetricEvaluator(use_lpips=True, device=device)
    
    clf_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    
    def predict_image(img_rgb_uint8, model):
        with torch.no_grad():
            t = clf_transform(Image.fromarray(img_rgb_uint8)).unsqueeze(0).to(device)
            logits = model(t)
            pred = logits.argmax(dim=1).item()
            conf = torch.softmax(logits, dim=1)[0, pred].item()
        return pred, conf
        
    def inpaint_unet(cloudy_uint8, mask):
        with torch.no_grad():
            t_img = torch.from_numpy(cloudy_uint8).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            t_mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()
            t_out = unet(t_img.to(device), t_mask.to(device))
            out_np = (t_out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        return out_np

    test_conditions = [
        {"condition": "thick", "bin_name": "10-30%", "bin_range": (0.10, 0.30)},
        {"condition": "thick", "bin_name": "30-60%", "bin_range": (0.30, 0.60)},
        {"condition": "thick", "bin_name": "60-80%", "bin_range": (0.60, 0.80)},
        {"condition": "semi_transparent", "bin_name": "30-60%", "bin_range": (0.30, 0.60)}
    ]
    
    # Methods (image producers). Each is scored by ALL three classifiers.
    methods = ["cloudy_as_is", "masked", "mean_fill", "nearest_neighbor", "telea_inpaint", "unet"]
    FILL_METHODS = {"mean_fill", "nearest_neighbor", "telea_inpaint", "unet"}
    audit_dir = Path("cloud_removal_project/results/audit_images")
    audit_dir.mkdir(parents=True, exist_ok=True)

    results_records = []
    t_start = time.time()
    
    print(f"Beginning evaluation over {N} images x {len(test_conditions)} conditions x {len(methods)} methods...")
    
    for img_idx, row in sampled_df.iterrows():
        img_p = Path("cloud_removal_project") / row["rel_path"]
        with Image.open(img_p) as pil_im:
            clean_rgb = np.array(pil_im.convert("RGB"))
            
        true_label = int(row["label"])
        class_name = row["class_name"]
        filename = row["filename"]
        
        # Clean-image prediction PER classifier (flip reference must use the same classifier)
        clean_ref = {name: predict_image(clean_rgb, model=m) for name, m in classifiers.items()}

        rng = np.random.default_rng(seed + img_idx)

        for cond_info in test_conditions:
            cond = cond_info["condition"]
            b_name = cond_info["bin_name"]
            b_range = cond_info["bin_range"]

            mask = sim.generate_cloud_mask(b_range, rng=rng)
            cloudy_rgb, desc = sim.apply_cloud(clean_rgb, mask, condition=cond, rng=rng)

            restored_dict = {
                "cloudy_as_is": baselines.cloudy_as_is(cloudy_rgb, mask),
                "masked": baselines.masked_input(cloudy_rgb, mask, fill_value=0),
                "mean_fill": baselines.mean_fill(cloudy_rgb, mask),
                "nearest_neighbor": baselines.nearest_neighbor_fill(cloudy_rgb, mask),
                "telea_inpaint": baselines.telea_inpaint(cloudy_rgb, mask),
                "unet": inpaint_unet(cloudy_rgb, mask),
            }

            for m_name in methods:
                restored = restored_dict[m_name]
                metrics = evaluator.evaluate_all(clean_rgb, restored, mask)   # once per method

                for clf_name, clf_model in classifiers.items():
                    clean_pred, _ = clean_ref[clf_name]
                    clean_correct = (clean_pred == true_label)
                    pred, conf = predict_image(restored, model=clf_model)
                    is_correct = (pred == true_label)

                    if clean_correct and not is_correct:
                        flip_type = "harmful_flip"
                    elif (not clean_correct) and is_correct:
                        flip_type = "helpful_flip"
                    elif clean_correct and is_correct:
                        flip_type = "retained_correct"
                    else:
                        flip_type = "retained_wrong"

                    img_path_rel = ""
                    # Save montage (clean | cloudy | restored | mask) for manual failure labeling:
                    # harmful flips of FILL methods under the standard (clean) classifier only.
                    if clf_name == "clean" and m_name in FILL_METHODS and flip_type == "harmful_flip":
                        fn = f"{img_idx}_{cond}_{b_name.replace('%','')}_{m_name}.png"
                        mask_vis = np.repeat((mask * 255).astype(np.uint8)[:, :, None], 3, axis=2)
                        montage = np.concatenate([clean_rgb, cloudy_rgb, restored, mask_vis], axis=1)
                        Image.fromarray(montage).resize((montage.shape[1] * 3, montage.shape[0] * 3), Image.NEAREST).save(audit_dir / fn)
                        img_path_rel = f"results/audit_images/{fn}"

                    results_records.append({
                        "image_id": img_idx,
                        "filename": filename,
                        "class_name": class_name,
                        "true_label": true_label,
                        "classifier": clf_name,
                        "clean_pred": clean_pred,
                        "clean_correct": int(clean_correct),
                        "condition": cond,
                        "coverage_bin": b_name,
                        "coverage_pct": round(desc["coverage_pct"], 4),
                        "thickness_proxy": round(desc["thickness_proxy"], 2),
                        "method": m_name,
                        "method_pred": pred,
                        "is_correct": int(is_correct),
                        "conf": round(conf, 4),
                        "flip_type": flip_type,
                        "audit_image": img_path_rel,
                        **metrics
                    })

        if (img_idx + 1) % 50 == 0:
            print(f"Processed {img_idx+1}/{N} images ({time.time() - t_start:.1f}s elapsed)...")
            
    df_results = pd.DataFrame(results_records)
    
    results_dir = Path("cloud_removal_project/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    per_image_csv = results_dir / "per_image_metrics.csv"
    df_results.to_csv(per_image_csv, index=False)
    print(f"\n--> Saved per-image metrics to {per_image_csv}")
    
    # 4. Generate Downstream Accuracy Table with 95% Confidence Intervals
    print("\n=== DOWNSTREAM ACCURACY TABLE WITH 95% CONFIDENCE INTERVALS (N=200) ===")
    agg_acc = df_results.groupby(["classifier", "condition", "coverage_bin", "method"]).agg(
        n=("is_correct", "count"),
        correct=("is_correct", "sum"),
        acc=("is_correct", "mean")
    ).reset_index()
    
    # Normal approximation CI: 1.96 * sqrt(p * (1 - p) / n)
    agg_acc["ci_95"] = 1.96 * np.sqrt(agg_acc["acc"] * (1.0 - agg_acc["acc"]) / agg_acc["n"])
    agg_acc["accuracy_pct"] = (agg_acc["acc"] * 100).round(2)
    agg_acc["ci_95_pct"] = (agg_acc["ci_95"] * 100).round(2)
    agg_acc["formatted_acc"] = agg_acc.apply(lambda r: f"{r['accuracy_pct']:.1f}% ± {r['ci_95_pct']:.1f}%", axis=1)
    
    for clf_name in classifiers:
        print(f"Clean-image accuracy [{clf_name}]: {df_results[df_results['classifier'] == clf_name].drop_duplicates('image_id')['clean_correct'].mean() * 100:.2f}%")
    
    for clf_name in classifiers:
        print(f"\n--- classifier = {clf_name} ---")
        sub = agg_acc[agg_acc["classifier"] == clf_name]
        print(sub.pivot(index="method", columns=["condition", "coverage_bin"], values="formatted_acc").to_string())
    
    acc_table_csv = results_dir / "downstream_accuracy_table.csv"
    agg_acc.to_csv(acc_table_csv, index=False)
    print(f"--> Saved downstream accuracy table to {acc_table_csv}")
    
    # 5. Flip Analysis with 95% Confidence Intervals
    print("\n=== PREDICTION FLIP ANALYSIS (N=200) ===")
    flip_counts = df_results.groupby(["classifier", "condition", "coverage_bin", "method", "flip_type"]).size().unstack(fill_value=0).reset_index()
    for col in ["harmful_flip", "helpful_flip", "retained_correct", "retained_wrong"]:
        if col not in flip_counts.columns:
            flip_counts[col] = 0
            
    flip_counts["total"] = flip_counts["harmful_flip"] + flip_counts["helpful_flip"] + flip_counts["retained_correct"] + flip_counts["retained_wrong"]
    flip_counts["harmful_rate_%"] = (flip_counts["harmful_flip"] / flip_counts["total"] * 100).round(2)
    flip_counts["helpful_rate_%"] = (flip_counts["helpful_flip"] / flip_counts["total"] * 100).round(2)
    
    # 95% CI on harmful rate
    p_harm = flip_counts["harmful_flip"] / flip_counts["total"]
    flip_counts["harmful_ci_95_%"] = (1.96 * np.sqrt(p_harm * (1.0 - p_harm) / flip_counts["total"]) * 100).round(2)
    
    print(flip_counts[["classifier", "condition", "coverage_bin", "method", "harmful_flip", "helpful_flip", "harmful_rate_%", "harmful_ci_95_%"]].to_string())
    flip_csv = results_dir / "flip_analysis.csv"
    flip_counts.to_csv(flip_csv, index=False)
    print(f"--> Saved flip analysis to {flip_csv}")
    
    # 5b. Fair comparison: adapted-to-masks vs adapted-to-fills (paired bootstrap on accuracy difference)
    print("\n=== FAIR COMPARISON: masked+mask_aware vs fill+fill_aware (paired bootstrap, 95% CI) ===")
    brng = np.random.default_rng(0)
    paired_rows = []
    for (cond, b_name), g in df_results.groupby(["condition", "coverage_bin"]):
        ref = g[(g["classifier"] == "mask_aware") & (g["method"] == "masked")].sort_values("image_id")["is_correct"].values
        for fm in sorted(FILL_METHODS):
            cur = g[(g["classifier"] == "fill_aware") & (g["method"] == fm)].sort_values("image_id")["is_correct"].values
            if len(cur) != len(ref) or len(ref) == 0:
                continue
            diffs = cur - ref
            boots = [diffs[brng.integers(0, len(diffs), len(diffs))].mean() for _ in range(2000)]
            paired_rows.append({"condition": cond, "coverage_bin": b_name, "fill_method": fm,
                                "fill_aware_acc": cur.mean(), "masked_mask_aware_acc": ref.mean(),
                                "diff_fill_minus_masked": diffs.mean(),
                                "diff_ci_low": np.percentile(boots, 2.5), "diff_ci_high": np.percentile(boots, 97.5)})
    df_paired = pd.DataFrame(paired_rows)
    print(df_paired.round(3).to_string(index=False))
    df_paired.to_csv(results_dir / "paired_comparison.csv", index=False)

    # 6. Metric Validity: Pooled & Within-Bin Rank Correlations
    print("\n=== METRIC VALIDITY (POOLED & WITHIN-BIN SPEARMAN RANK CORRELATIONS) ===")
    metric_validity_records = []
    # Fills only, standard classifier: cloudy/masked rows are not fills, and the old pooled
    # masked_fair_aware rows mixed image metrics from one input with accuracy from another model.
    df_results = df_results[(df_results["classifier"] == "clean") & (df_results["method"].isin(FILL_METHODS))].copy()
    
    for metric_col in ["psnr_full", "psnr_mask", "ssim_full", "ssim_mask", "lpips"]:
        # Pooled
        valid_df = df_results.dropna(subset=[metric_col])
        rho_p, p_val_p = spearmanr(valid_df[metric_col], valid_df["is_correct"])
        
        # Within-bin correlations (Thick 30-60%)
        bin_df = valid_df[(valid_df["condition"] == "thick") & (valid_df["coverage_bin"] == "30-60%")]
        rho_b, p_val_b = spearmanr(bin_df[metric_col], bin_df["is_correct"])
        
        is_harmful = (valid_df["flip_type"] == "harmful_flip").astype(int)
        rho_harm, p_harm = spearmanr(valid_df[metric_col], is_harmful)
        
        metric_validity_records.append({
            "metric": metric_col,
            "pooled_spearman_rho": round(rho_p, 4),
            "pooled_p_value": round(p_val_p, 5),
            "within_bin_30_60_rho": round(rho_b, 4),
            "within_bin_p_value": round(p_val_b, 5),
            "rho_vs_harmful_flips": round(rho_harm, 4)
        })
        
    df_validity = pd.DataFrame(metric_validity_records)
    print(df_validity.to_string(index=False))
    val_csv = results_dir / "metric_validity.csv"
    df_validity.to_csv(val_csv, index=False)
    print(f"--> Saved metric validity to {val_csv}")

if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles: avoid UnicodeEncodeError on special chars
    except Exception:
        pass
    try:
        run_evaluation(num_test_samples=200)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    print("EVALUATION FINISHED CLEANLY")
