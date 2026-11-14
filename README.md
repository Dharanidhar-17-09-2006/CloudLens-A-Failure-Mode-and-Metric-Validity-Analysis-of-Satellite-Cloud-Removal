# CloudLens: When Does Filling Cloud-Covered Pixels Help Land-Cover Classification?

Failure-mode and metric-validity analysis of satellite cloud removal.
Course project, **AI61201 Visual Computing with AI/ML**, IIT Kharagpur.

- **Authors:** P. Sai Dharanidhar Ram (24AI10004), Ruthvik Devarakonda (24AI10023)
- **Mentor:** Prof. Somdyuti Paul
- **Report:** IEEE-format PDF, submitted on Moodle (the Markdown working copy is `cloud_removal_project/PROJECT_REPORT_FINAL.md`)

## What this project asks

When part of a satellite tile is hidden by cloud, should we leave the cloud in, mask it, fill it, or fill it and retrain the classifier on the filled tiles? We compare these options on EuroSAT (synthetic clouds) and RICE (real clouds), and test whether PSNR, SSIM and LPIPS warn us when a fill misleads the classifier. The evaluation rules were written in `ANALYSIS_PLAN.md` before the full multi-seed runs.

### Main results (from the report)

| Finding | Result |
|---|---|
| Fill-aware classifier vs mask-aware classifier (best fill, accuracy gain) | +5.2%, +8.3%, +13.2% at 10–30%, 30–60%, 60–80% cloud coverage |
| Telea vs scratch U-Net at 60–80% coverage | 0.830 vs 0.826 accuracy (comparable) |
| PSNR, SSIM, LPIPS vs harmful flips (Spearman ρ) | pooled \|ρ\| ≤ 0.22; within-method up to 0.24 |
| Opacity sweep (clean classifier) | leaving thin cloud in is better than filling below α ≈ 0.65 |

All accuracy numbers use the full 4,050-tile EuroSAT test split, mean over 3 seeds.

---

## DEMO / INFERENCE (start here)

**`CloudLens_Demo_Final.ipynb` is the demonstration and inference notebook.** It runs in inference mode only: no training. It loads trained weights and stored experiment outputs, re-runs inference on test tiles, and prints the report tables and figures.

### What the notebook needs

One folder, `cloudlens_assets`, with three subfolders. It is provided as **`cloudlens_assets.zip`** (Moodle upload and Drive link below):

| Folder | Contents |
|---|---|
| `models/` | `resnet18_clean.pth`, `resnet18_mask_aware.pth`, `resnet18_fill_aware.pth`, `unet_inpaint.pth`, and the ResNet-encoder U-Net checkpoints for the transfer-learning check |
| `data/` | **Sample data:** `splits/test.csv` and the 4,050 EuroSAT test tiles (64×64 RGB), at the paths the notebook expects |
| `results/` | Stored experiment outputs (CSV files) that the notebook displays: three-seed accuracy, metric-validity tiles, opacity sweep, RICE evaluation, manual failure audit, transfer learning |

Links :
- Colab notebook: `https://colab.research.google.com/drive/1JLVZQq4ZisyiZscREWG-ItBilwyWqkNM`
- Assets ZIP: `https://drive.google.com/file/d/13w8ELUKSMXNHJdt99O8h09e-9Nkg4Cuc/view?usp=drive_link` (the Drive file ID is already configured in Section 0; the ZIP is also uploaded separately on Moodle)

### Run on Google Colab (recommended)

1. Open the Colab notebook link above. If Colab disables **Runtime → Run all** because the notebook is shared for editing, first choose **File → Save a copy in Drive**, then open and run your independent copy.
2. The Google Drive file ID for `cloudlens_assets.zip` is **already configured** as `ASSETS_GDRIVE_ID` in the configuration cell (Section 0). You do not need to edit this value or manually extract the ZIP.
3. Select **Runtime → Run all**. The notebook automatically downloads `cloudlens_assets.zip`, validates it, extracts it, and locates the required assets.
4. **If the automatic Drive download fails:** run the notebook's configuration cell. It will prompt you to upload `cloudlens_assets.zip`. Upload the ZIP when prompted; the notebook will validate and extract it and locate the required assets automatically. The ZIP is also provided separately on Moodle as a fallback.
5. A GPU runtime is recommended for faster execution. The notebook also supports CPU execution, although some live inference steps may take longer.

### Run on Kaggle

1. Upload `cloudlens_assets.zip` as a Kaggle dataset and attach it to the notebook.
2. Import `CloudLens_Demo_Final.ipynb`, then *Run all*. The notebook searches `/kaggle/input` for the assets.

### Run locally

```bash
pip install torch torchvision numpy pandas scipy scikit-image opencv-python-headless matplotlib Pillow lpips jupyter gdown
unzip cloudlens_assets.zip -d cloudlens_assets      # next to the notebook
jupyter notebook CloudLens_Demo_Final.ipynb
```

### What you will see

| Notebook section | Output | Source |
|---|---|---|
| 3 | Table I: accuracy by coverage (mean ± SD, 3 seeds), decision-rule check, bar chart | stored `threeway_seed*.csv` |
| 4 | Live re-evaluation of the clean, mask-aware and fill-aware classifiers with the four fills | **inference** with saved weights |
| 5 | Tables IV–VI: Spearman ρ with bootstrap CIs, within-method ρ, harmful-flip rates | stored per-tile CSV |
| 6 | Table II: opacity sweep and dehazing (dark channel prior) | **inference** with `resnet18_clean.pth` |
| 7 | Failure taxonomy (manual audit) and water-body collapse | stored audit CSVs |
| 8 | Table III: real clouds (RICE) | stored RICE evaluation CSVs |
| 9 | Table VII: transfer learning | stored CSV plus live check with saved U-Nets |
| 10 | Qualitative strips: clean, cloudy, masked, four fills, with predictions | **inference** |
| 11 | Report values vs values computed in the notebook | summary table |

Settings in the config cell: `N_LIVE` (test tiles for the live re-evaluation, 4,050 on GPU or 1,000 on CPU by default), `N_BOOT` (bootstrap repetitions, 200 here, 1,000 in the report), `N_OPACITY` (500).

### Why live numbers differ slightly from Table I

Table I is the mean over three seeds. Those classifiers were trained inside `threeway_comparison.py` and were not saved. The checkpoints in `models/` come from an earlier single-seed run (`train_mask_aware_classifier.py`, `train_fill_classifier.py`), where only the last ResNet blocks and the head are fine-tuned. So the live section (4) reproduces the ranking and the decision rule, but its numbers can differ by a few points from Table I.

---

## TRAINING AND EXPERIMENT CODE

The model checkpoints, datasets and result files are not in the repository (see `.gitignore`). The commands below regenerate them. Run every command from the **repository root**, the folder that contains `cloud_removal_project/`. A GPU is recommended for `train_unet.py`, `threeway_comparison.py`, `unet_transfer.py` and the RICE training scripts.

### 1. Environment

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch torchvision numpy pandas scipy scikit-image opencv-python-headless matplotlib Pillow lpips
```

### 2. Data

**EuroSAT** (27,000 tiles, 64×64, 10 classes). Download with torchvision:

```bash
python -c "from torchvision.datasets import EuroSAT; EuroSAT(root='cloud_removal_project/data/eurosat', download=True)"
```

The tiles end up in `cloud_removal_project/data/eurosat/eurosat/2750/<ClassName>/*.jpg`.

**RICE** (real paired clouds, Lin et al., arXiv:1901.00600). Download the dataset and place `RICE1` and `RICE2`, each with `cloud/` and `label/` subfolders, under `cloud_removal_project/data/RICE_DATASET/`. This is only needed for the RICE experiments (step 8).

### 3. Pipeline, in order

| Step | Command | Writes | Report item |
|---|---|---|---|
| 1 | `python cloud_removal_project/src/audit_and_split.py` | `data/splits/{train,val,test}.csv` (70/15/15, seed 42), `AUDIT_NOTES.md` | Dataset split |
| 2 | `python cloud_removal_project/src/classifier.py` | `models/resnet18_clean.pth` | Clean classifier, opacity sweep |
| 3 | `python cloud_removal_project/src/train_unet.py` | `models/unet_inpaint.pth` (30 epochs, GPU) | Scratch U-Net fill |
| 4 | `python cloud_removal_project/src/threeway_comparison.py --seed 0` (repeat with `--seed 1` and `--seed 2`) | `results/threeway_seed{0,1,2}.csv` | **Table I**, main result |
| 5 | `python cloud_removal_project/src/metric_validity_full.py --seed 0`, then `summarize_metric_validity_tiles.py --seed 0`, then `within_method_validity.py --seed 0` | `results/metric_validity_tiles_seed0.csv` and summaries | **Tables IV, V, VI** |
| 6 | `python cloud_removal_project/src/opacity_sweep.py` | `results/opacity_sweep_v2.csv` and plot | **Table II** |
| 7 | `python cloud_removal_project/src/train_mask_aware_classifier.py`, `train_fill_classifier.py`, `evaluate_all.py`, then `audit_failures.py` | demo checkpoints, `results/prediction_shift_fills.csv`, `manual_label_sheet.csv` | Failure taxonomy |
| 8 | RICE: see below | `results/rice_inpaint/eval/*.csv` | **Table III** |
| 9 | `python cloud_removal_project/src/unet_transfer.py --seed 0 --imagenet_lr 1e-4` | `models/unet_transfer_*`, `results/unet_transfer_seed0*.csv` | **Table VII** |
| 10 | `python cloud_removal_project/src/generate_figures.py` | `results/figures/` | Figures |

Order matters: train the U-Net (step 3) before steps 4, 5 and 7, which load `models/unet_inpaint.pth`.

**Step 7, manual audit.** `audit_failures.py` writes `results/manual_label_sheet.csv` with a montage image for each sampled harmful flip. Fill the `manual_label` column by hand from the montages, then run `audit_failures.py` again to produce the taxonomy summary.

**Step 8, RICE experiments:**

```bash
python cloud_removal_project/src/train_rice_inpainter.py --name small --arch small
python cloud_removal_project/src/train_rice_inpainter.py --name big --arch big
python cloud_removal_project/src/train_rice_inpainter.py --name big_gan --arch big --gan 1
python cloud_removal_project/src/train_paired_restorer.py --name small --arch small
python cloud_removal_project/src/train_paired_restorer.py --name big --arch big
python cloud_removal_project/src/eval_rice_inpainters.py
python cloud_removal_project/src/eval_rice_inpainters.py --dilate 3     # dilated-mask check
```

### Reproducibility notes

- Splits use a fixed seed (42). Test clouds are generated with fixed per-tile seeds, so they are identical across runs and seeds.
- The three seeds change classifier training only (initialisation of the head, data order, the filled training set). The scratch U-Net is trained once and shared.
- Classifier training uses ImageNet-initialised ResNet-18, 3 epochs, Adam (lr 3e-4), batch size 128, with the epoch chosen by validation accuracy (`threeway_comparison.py`).
- Training masks (random ellipses and rectangles) differ from test masks (multi-scale noise clouds), so classifiers never see test-style cloud shapes. The scratch U-Net is trained on the noise-based clouds.

---

## FILE GUIDE

### Repository root

| File | Purpose |
|---|---|
| `CloudLens_Demo_Final.ipynb` | **Demo and inference notebook** (see above) |
| `ANALYSIS_PLAN.md` | Evaluation protocol and decision rule, written before the full multi-seed runs |
| `generate_slide_assets.py` | Draws the figures used in the presentation |
| `slide_assets/` | Figures used in the presentation |
| `patch_scene_cap.py` | One-time patch that added scene-cap options to `build_sen12mscr_subset.py` (not needed to reproduce results) |
| `archive_old/` | Superseded report drafts and intermediate outputs (not needed) |

### `cloud_removal_project/src/` (core)

| File | Purpose |
|---|---|
| `audit_and_split.py` | Audits EuroSAT and writes the stratified train/val/test splits |
| `dataset.py` | EuroSAT dataset loader |
| `cloud_simulation.py` | Synthetic clouds: noise-based test clouds, ellipse/rectangle training masks, thick and semi-transparent rendering |
| `inpainting_baselines.py` | Cloudy as-is, black mask, mean fill, nearest neighbour, Telea |
| `dehaze_dcp.py` | Dark channel prior dehazing |
| `unet_inpaint.py`, `train_unet.py` | Small 4-channel U-Net inpainter and its training |
| `classifier.py` | ResNet-18 classifier and training of the clean classifier |
| `metrics.py` | PSNR, SSIM and LPIPS, whole tile and inside the cloud mask |

### `cloud_removal_project/src/` (experiments)

| File | Purpose |
|---|---|
| `ablate_classifier_finetune.py` | Shared training and evaluation code for the classifier regimes (also a fine-tuning ablation) |
| `threeway_comparison.py` | **Table I**: clean, mask-aware and fill-aware classifiers on all inputs, one seed per run |
| `train_mask_aware_classifier.py`, `train_fill_classifier.py` | Earlier single-seed mask-aware and fill-aware classifiers (the demo checkpoints) |
| `evaluate_all.py` | Earlier full benchmarking pipeline with confidence intervals |
| `metric_validity_full.py` | Per-tile PSNR/SSIM/LPIPS and harmful flips on the full test split |
| `summarize_metric_validity_tiles.py` | Tile-level bootstrap for **Table IV** |
| `within_method_validity.py` | Within-method correlations, **Table V** |
| `opacity_sweep.py` | **Table II**: opacity sweep and dehazing |
| `audit_failures.py`, `plot_taxonomy.py` | Prediction-shift analysis, manual audit sheet, taxonomy plot |
| `unet_transfer.py` | **Table VII**: ResNet-encoder U-Nets (random, ImageNet, satellite pretraining) |
| `generate_figures.py` | Result figures |

### `cloud_removal_project/src/` (real clouds, RICE)

| File | Purpose |
|---|---|
| `rice_data.py`, `rice_models.py` | RICE loading and splits; larger generator and PatchGAN discriminator |
| `train_rice_inpainter.py`, `train_paired_restorer.py` | Train RICE inpainters and supervised paired restorers |
| `eval_rice_inpainters.py` | **Table III**: LPIPS and classifier agreement on held-out RICE tiles |
| `real_cloud_eval.py`, `real_fidelity_eval.py`, `real_diagnostics.py` | Earlier real-cloud evaluation and classifier-stability diagnostics |

### Optional, not used in the reported results

| File | Purpose |
|---|---|
| `build_sen12mscr_subset.py`, `sen12_threeway.py` | Experimental labelled real-cloud test on SEN12MS-CR (requires its own data download) |

---

## Limitations

EuroSAT clouds are synthetic, flat and white, on 64×64 RGB tiles, with no cirrus, shadows or infrared bands. RICE has no land-cover labels, so real-cloud results use classifier agreement as a proxy. The fill-aware classifier is trained on the same fill methods it is tested on.

## Citation

If you use this code, please cite the project report (CloudLens, AI61201, IIT Kharagpur, 2026).
