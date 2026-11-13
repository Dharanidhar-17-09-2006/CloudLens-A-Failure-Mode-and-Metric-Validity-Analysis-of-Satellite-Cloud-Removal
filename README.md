# Cloud Removal Failure-Mode Analysis

A controlled failure-mode and downstream utility study evaluating satellite image cloud inpainting on land-cover classification.

## Project Structure
- `data/`: EuroSAT dataset and reproducible stratified train/val/test splits (`seed=42`).
- `src/`:
  - `audit_and_split.py`: EuroSAT auditing and stratified split generator.
  - `cloud_simulation.py`: Multi-scale organic cloud mask generation and physical corruption simulation.
  - `metrics.py`: Full and in-mask PSNR, SSIM, and LPIPS perceptual distance computation.
  - `inpainting_baselines.py`: Classical and simple fill baselines (Cloudy As-Is, Zero-mask, Mean fill, Nearest-neighbor, Telea).
  - `unet_inpaint.py` & `train_unet.py`: Lightweight 4-channel PyTorch U-Net inpainter.
  - `classifier.py`: Downstream ResNet-18 classifier (clean baseline, 94.77% val accuracy).
  - `train_mask_aware_classifier.py`: Fair Mask-Aware Classifier fine-tuned with missing regions.
  - `evaluate_all.py`: Comprehensive benchmarking pipeline across all 7 methods, coverage bins, and conditions with 95% CIs.
  - `audit_failures.py`: Empirical failure taxonomy and 100-sample audit generator.
  - `generate_figures.py`: Generates publication-ready figures in `results/figures/`.
- `models/`: Saved PyTorch model checkpoints (`resnet18_clean.pth`, `resnet18_mask_aware.pth`, `unet_inpaint.pth`).
- `results/`: Per-image CSVs, downstream accuracy matrix, flip analysis, metric validity table, 100-sample failure audit, and visual figures.
- `PROJECT_REPORT.md`: Full scientific report, empirical failure taxonomy, and statistical findings.
- `SLIDES.md`: 9-slide presentation deck.

## How to Run (run from the folder that CONTAINS `cloud_removal_project/`)
```bash
python cloud_removal_project/src/audit_and_split.py
python cloud_removal_project/src/classifier.py                    # clean classifier
python cloud_removal_project/src/train_unet.py                    # full data, 30 epochs, best-val checkpoint (GPU recommended)
python cloud_removal_project/src/train_mask_aware_classifier.py   # adapted to masks (held-out shape-mask family)
python cloud_removal_project/src/train_fill_classifier.py         # adapted to fills (same held-out masks)
python cloud_removal_project/src/evaluate_all.py                  # must print EVALUATION FINISHED CLEANLY
python cloud_removal_project/src/audit_failures.py                # prediction-shift table + manual_label_sheet.csv
#   -> open results/manual_label_sheet.csv, fill `manual_label` from the montage images, re-run audit_failures.py
python cloud_removal_project/src/generate_figures.py
```
Order matters: U-Net before the fill classifier (it trains on U-Net fills if the checkpoint exists).
Old v1 outputs are in `results_v1_archive/` (stale: unfair comparison, rule-based taxonomy).
