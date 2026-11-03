> **STATUS: numbers below are v1 (pre-fix). Regenerate after running the v2 pipeline - see REVISION_NOTES.md.**

# Cloud Removal Failure-Mode Analysis: Empirical Research Report

**Course**: Visual Computing with AI/ML (AI61201)  
**Team**: P. Sai Dharanidhar Ram (24AI10004), Ruthvik Devarakonda (24AI10023)  
**Mentor**: Prof. Somdyuti Paul, Department of Artificial Intelligence, IIT Kharagpur  

---

## 1. Executive Summary & Research Context

Following guidance from Professor Somdyuti Paul, this study evaluates whether inpainting cloud-covered satellite pixels adds utilitarian value to downstream land-cover classification or actively misleads it. Inpainting is restricted strictly to complete-tile pipeline constraints, excluding defense, weather, and disaster response.

We evaluate **7 restoration and masking conditions** across **3 thick opaque cloud bins** (10–30%, 30–60%, 60–80%) and a **secondary semi-transparent condition** (30–60% coverage at 50% opacity) on the Sentinel-2 EuroSAT benchmark. All results report sample size ($N=200$ test images per condition, 5,600 total evaluations) and 95% confidence intervals.

---

## 2. Experimental Setup & Protocol

- **Dataset**: EuroSAT (27,000 Sentinel-2 64x64 RGB tiles, 10 balanced land-cover classes).
- **Splits**: Stratified random split (`seed=42`): Train (70%, 18,900), Validation (15%, 4,050), Test (15%, 4,050).
- **Classifiers**:
  1. *Clean Benchmark Classifier*: ResNet-18 trained on clean EuroSAT (Validation Accuracy: **94.77%**, Test Accuracy: **97.50% ± 2.2%**).
  2. *Fair Mask-Aware Classifier*: ResNet-18 fine-tuned with random missing regions (10–80%), learning to classify visible pixels without hallucinating.
- **Methods Evaluated**:
  1. `cloudy_as_is`: Raw corrupted cloudy tile evaluated on Clean Classifier.
  2. `masked_unadapted`: Zero-masked tile evaluated on Clean Classifier (which has never seen black masks).
  3. `masked_fair_aware`: Zero-masked tile evaluated on the **Mask-Aware Classifier**.
  4. `mean_fill`: Cloud region replaced with mean RGB color of visible context.
  5. `nearest_neighbor`: Euclidean distance transform spatial propagation.
  6. `telea_inpaint`: Alexandru Telea Fast Marching Method (`cv2.inpaint`).
  7. `unet_prototype`: Lightweight 4-channel PyTorch U-Net prototype (time-boxed 2 epochs CPU training).

---

## 3. Downstream Classification Accuracy Matrix

*Sample size $N=200$ per bin. Reported with 95% confidence intervals ($\pm 1.96 \sqrt{p(1-p)/N}$). Clean Benchmark: 97.50% ± 2.2%.*

| Restoration Strategy | Semi-Transparent (30–60%) | Thick Clouds (10–30%) | Thick Clouds (30–60%) | Thick Clouds (60–80%) |
|---|---|---|---|---|
| **Cloudy As-Is** | 62.0% ± 6.7% | 71.0% ± 6.3% | 43.0% ± 6.9% | 17.0% ± 5.2% |
| **Masked (Unadapted)** | 20.5% ± 5.6% | 44.5% ± 6.9% | 20.5% ± 5.6% | 14.0% ± 4.8% |
| **Mask-Aware (Fair Baseline)** | **79.5% ± 5.6%** | **88.5% ± 4.4%** | **85.0% ± 5.0%** | **65.0% ± 6.6%** |
| **Mean Fill** | 50.0% ± 6.9% | 87.5% ± 4.6% | 52.5% ± 6.9% | 13.0% ± 4.7% |
| **Nearest Neighbor** | 77.0% ± 5.8% | **94.0% ± 3.3%** | 81.5% ± 5.4% | 40.5% ± 6.8% |
| **Telea Inpaint** | 61.5% ± 6.7% | 91.5% ± 3.9% | 57.5% ± 6.8% | 23.0% ± 5.8% |
| **Prototype U-Net (2-ep)** | 38.0% ± 6.7% | 89.0% ± 4.3% | 63.0% ± 6.7% | 17.5% ± 5.3% |

### Core Findings on Downstream Accuracy:
1. **Light Clouds (10–30%)**: Infilling provides practical value. Nearest Neighbor achieves **94.0% ± 3.3%**, Telea reaches **91.5% ± 3.9%**, and Prototype U-Net reaches **89.0% ± 4.3%**, all outperforming cloudy as-is (71.0%) and matching or exceeding fair masking (88.5%).
2. **Moderate Clouds (30–60%)**: Nearest Neighbor spatial infill achieves **81.5% ± 5.4%**, outperforming the CPU-prototype U-Net (63.0%) and Telea (57.5%). However, **the Fair Mask-Aware Classifier achieves 85.0% ± 5.0%**, demonstrating that adapting the downstream model to ignore missing pixels beats infilling.
3. **Heavy Clouds (60–80%)**: Classical inpainting and prototype U-Net degrade sharply to **17.5–23.0%**, offering no meaningful benefit over raw cloudy tiles (17.0%). While Nearest Neighbor retains **40.5% ± 6.8%**, the **Fair Mask-Aware Classifier decisively outperforms all inpainting at 65.0% ± 6.6%**.

---

## 4. Failure Analysis & Prediction Flips

We classify prediction transitions against clean predictions into:
- **Harmful Flip**: Clean image was correctly classified $\rightarrow$ infilled image is misclassified (hallucinated error).
- **Helpful Flip**: Cloudy image was misclassified $\rightarrow$ infilled image recovers correct class.

*Table: Harmful Flip Counts and Rates (N=200 per condition, with 95% CI)*

| Condition | Coverage Bin | Method | Harmful Flips | Harmful Rate (%) | 95% CI | Helpful Flips |
|---|---|---|---|---|---|---|
| **Thick** | 10–30% | Cloudy As-Is | 53 | 26.5% | ± 6.1% | 0 |
| **Thick** | 10–30% | Nearest Neighbor | 8 | **4.0%** | ± 2.7% | 1 |
| **Thick** | 10–30% | Telea | 13 | 6.5% | ± 3.4% | 1 |
| **Thick** | 10–30% | Mask-Aware Fair | 18 | 9.0% | ± 4.0% | 0 |
| **Thick** | 10–30% | Prototype U-Net | 18 | 9.0% | ± 4.0% | 1 |
| **Thick** | 30–60% | Cloudy As-Is | 110 | 55.0% | ± 6.9% | 1 |
| **Thick** | 30–60% | Mask-Aware Fair | 27 | **13.5%** | ± 4.7% | 2 |
| **Thick** | 30–60% | Nearest Neighbor | 33 | 16.5% | ± 5.1% | 1 |
| **Thick** | 30–60% | Prototype U-Net | 70 | 35.0% | ± 6.6% | 1 |
| **Thick** | 30–60% | Telea | 82 | 41.0% | ± 6.8% | 2 |
| **Thick** | 60–80% | Mask-Aware Fair | 66 | **33.0%** | ± 6.5% | 1 |
| **Thick** | 60–80% | Nearest Neighbor | 114 | 57.0% | ± 6.9% | 0 |
| **Thick** | 60–80% | Telea | 149 | 74.5% | ± 6.0% | 0 |
| **Thick** | 60–80% | Prototype U-Net | 160 | 80.0% | ± 5.5% | 0 |
| **Thick** | 60–80% | Mean Fill | 169 | 84.5% | ± 5.0% | 0 |

---

## 5. Data-Driven Empirical Failure Taxonomy

Analysis of all 2,384 harmful flips across the dataset reveals the exact class transitions:

1. **Water Body / Dark Infill Collapse (`SeaLake` / `River` shift) — 64.5% (1,538 cases)**:
   - When large regions are zeroed, mean-filled, or smoothed with low reflectance, CNN feature maps interpret the absence of high-frequency texture as open water. 
   - Over 64% of all harmful errors misclassify terrestrial classes (`Highway`, `Industrial`, `Residential`, `Forest`) into `SeaLake` (1,113 cases) or `River` (409 cases).
2. **Vegetative Texture Smear (`AnnualCrop`, `PermanentCrop`, `Pasture`) — 16.4% (392 cases)**:
   - Inpainting blends surrounding green pixels across borders, erasing sharp boundaries and causing infrastructure to be misclassified as homogeneous agricultural fields.
3. **Built Environment Misclassification (`Industrial`, `Residential`) — 13.6% (325 cases)**:
   - Fragmented or patchy infilling creates synthetic edge artifacts resembling complex urban boundaries.
4. **Contextual Boundary Error — 3.5% (83 cases)**:
   - Infilling over transitions (e.g., riverbanks, coastline) blurs the interface.
5. **Linear Infrastructure Discontinuity (`Highway`) — 1.9% (46 cases)**:
   - Narrow linear transport corridors are severed by spatial interpolation.

*A detailed audit of 100 representative failure instances is compiled in [`results/failure_sample_audit.csv`](file:///cloud_removal_project/results/failure_sample_audit.csv).*

---

## 6. Metric Validity Analysis

*Correlation with classification accuracy and harmful flips ($N=200$ images $\times$ 4 conditions $\times$ 7 methods = 5,600 evaluations)*:

| Metric | Pooled Spearman $\rho$ | $p$-value | Within-Bin (30–60%) $\rho$ | $p$-value | $\rho$ vs. Harmful Flips |
|---|---|---|---|---|---|
| **LPIPS** | **-0.3639** | $< 10^{-15}$ | **-0.2784** | $< 10^{-15}$ | **+0.3730** |
| **SSIM (Full)** | +0.3634 | $< 10^{-15}$ | +0.2344 | $< 10^{-15}$ | -0.3665 |
| **PSNR (Full)** | +0.2017 | $< 10^{-15}$ | +0.1587 | $< 10^{-15}$ | -0.1994 |
| **SSIM (Mask)** | +0.1334 | $< 10^{-15}$ | +0.1442 | $< 10^{-15}$ | -0.1329 |
| **PSNR (Mask)** | **+0.1242** | $< 10^{-15}$ | **+0.1389** | $< 10^{-15}$ | **-0.1209** |

### Insights on Metrics:
- **LPIPS and Full SSIM are competitive**: LPIPS ($\rho = -0.3639$) and Full SSIM ($\rho = +0.3634$) show virtually identical correlation with downstream performance, both pooled and within individual bins.
- **In-Mask PSNR is Weak**: In-mask PSNR ($\rho = 0.1242$) is poorly correlated with classifier success. Fills that achieve mathematically low MSE inside the mask frequently trigger water-body tone collapse.

---

## 7. Limitations & Honest Scope

1. **Synthetic Cloud Assumption**: Synthetic multi-scale white clouds on 64x64 EuroSAT RGB do not model atmospheric cirrus translucency, real cloud shadows, or multi-spectral NIR/SWIR bands.
2. **Prototype Deep Inpainter**: The U-Net was trained for 2 epochs on a CPU subset as a lightweight prototype. v2 retrains the U-Net properly; interpret the deep-vs-classical comparison from the v2 results.
3. **Model Weights Omitted from Zip**: Heavy `.pth` checkpoints (~52 MB) were excluded from the distributed code archive to keep the package under 1 MB, but can be reproduced in 3–5 minutes via the included training scripts.
