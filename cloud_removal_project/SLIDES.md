# Cloud Removal Failure-Mode Analysis: Presentation Deck

**Course**: Visual Computing with AI/ML (AI61201)  
**Team**: P. Sai Dharanidhar Ram (24AI10004), Ruthvik Devarakonda (24AI10023)  
**Instructor / Guide**: Prof. Somdyuti Paul, Dept. of Artificial Intelligence, IIT Kharagpur  

---

## Slide 1: Problem & Scope
### Does Cloud Inpainting Benefit Satellite Land-Cover Pipelines?
- **Practical Context**: Operational LULC pipelines require fixed-dimension rectangular tiles. Cloud gaps force a choice: drop the tile, zero-mask it, or inpaint it.
- **Scope Restriction**: Inpainted pixels are fabricated. We strictly exclude safety-critical domains (defense, weather forecasting, disaster response).
- **Goal**: Test whether inpainting improves classification over masking and simple fills, and establish where fabricated content misleads the classifier.

---

## Slide 2: Addressing Faculty Guidance
### Scope Refinements & Core Definitions
1. **Opaque Clouds vs. Dehazing**:
   - Thick clouds have 0% ground transmission. Inpainting is treated as a plausible visual infill, not surface recovery.
   - Semi-transparent clouds (50% opacity) are evaluated separately as a secondary condition.
2. **Fair Comparison**:
   - We introduce a **Mask-Aware Classifier** (fine-tuned with random missing regions) as a fair masked baseline alongside traditional clean-only models.
   - Flips are rigorously categorized into **Helpful** (fixing a cloudy error) vs. **Harmful** (hallucinated error on clean-correct tiles).

---

## Slide 3: Experimental Setup & Statistical Protocol
- **Dataset**: EuroSAT (Sentinel-2, 27,000 images, 10 classes, 64x64 RGB).
- **Splits**: Stratified split (`seed=42`): Train (70%), Val (15%), Test (15%).
- **Sample Size**: $N = 200$ test images per condition (5,600 total evaluations across 7 methods).
- **Statistical Rigor**: All accuracy and flip tables report 95% Confidence Intervals ($\pm 1.96 \sqrt{p(1-p)/N}$).
- **Models**:
  - Clean ResNet-18 Benchmark: **97.50% ± 2.2%** test accuracy.
  - Fair Mask-Aware ResNet-18: Fine-tuned on masked imagery.
  - Prototype U-Net: Lightweight 4-channel model (time-boxed 2 epochs CPU).

---

## Slide 4: Downstream Classification Accuracy Matrix
*Clean Benchmark: 97.50% ± 2.2% (N=200, 95% CI)*

| Restoration Strategy | Semi-Transparent (30–60%) | Thick Clouds (10–30%) | Thick Clouds (30–60%) | Thick Clouds (60–80%) |
|---|---|---|---|---|
| **Cloudy As-Is** | 62.0% ± 6.7% | 71.0% ± 6.3% | 43.0% ± 6.9% | 17.0% ± 5.2% |
| **Masked (Unadapted)** | 20.5% ± 5.6% | 44.5% ± 6.9% | 20.5% ± 5.6% | 14.0% ± 4.8% |
| **Mask-Aware (Fair)** | **79.5% ± 5.6%** | **88.5% ± 4.4%** | **85.0% ± 5.0%** | **65.0% ± 6.6%** |
| **Mean Fill** | 50.0% ± 6.9% | 87.5% ± 4.6% | 52.5% ± 6.9% | 13.0% ± 4.7% |
| **Nearest Neighbor** | 77.0% ± 5.8% | **94.0% ± 3.3%** | 81.5% ± 5.4% | 40.5% ± 6.8% |
| **Telea Inpaint** | 61.5% ± 6.7% | 91.5% ± 3.9% | 57.5% ± 6.8% | 23.0% ± 5.8% |
| **Prototype U-Net (2-ep)** | 38.0% ± 6.7% | 89.0% ± 4.3% | 63.0% ± 6.7% | 17.5% ± 5.3% |

---

## Slide 5: Key Takeaways on Fill vs. Mask
1. **Light Clouds (10–30%)**: Infilling provides practical value. Nearest Neighbor reaches **94.0%**, beating cloudy-as-is (71.0%) and matching fair masking (88.5%).
2. **Moderate Clouds (30–60%)**: Inpainting helps (Nearest Neighbor 81.5%), but **fine-tuning the downstream classifier to handle masks (85.0%) outperforms all inpainting**.
3. **Severe Clouds (60–80%)**: Classical inpainting and prototype U-Net collapse (17.5–23.0%). While Nearest Neighbor retains 40.5%, **the Mask-Aware Classifier achieves 65.0%**, (v1 number: compared against unadapted fills; to be re-checked with the fair comparison).

---

## Slide 6: Failure Analysis & Harmful Prediction Flips
*Harmful Flip = Clean image was correct $\rightarrow$ Infilled image caused misclassification*

- **10–30% Thick Clouds**: Low harmful flip rates across the board:
  - Nearest Neighbor: **4.0% ± 2.7%** | Telea: **6.5%** | Prototype U-Net: **9.0%** | Fair Mask: **9.0%**
- **30–60% Thick Clouds**:
  - Fair Mask: **13.5% ± 4.7%** | Nearest Neighbor: **16.5%** | Prototype U-Net: **35.0%** | Telea: **41.0%**
- **60–80% Thick Clouds**:
  - Fair Mask: **33.0% ± 6.5%** | Nearest Neighbor: **57.0%** | Telea: **74.5%** | Prototype U-Net: **80.0%**
- **Empirical Insight**: Above 60% cloud cover, generative and diffusion-like infills mislead the classifier in 75–80% of test tiles.

---

## Slide 7: Empirical Failure Taxonomy
*Derived directly from 2,384 harmful flip confusion transitions across all experiments:*

1. **Water Body / Dark Infill Collapse (64.5% of harmful flips)**:
   - Zero-masking, smooth interpolation, and dark infills produce flat, low-reflectance regions.
   - CNN feature maps interpret the lack of high-frequency texture as water, collapsing terrestrial classes into `SeaLake` (1,113 cases) and `River` (409 cases).
2. **Vegetative Texture Smear (16.4%)**:
   - Infill spreads green hue across boundaries, misclassifying built environment into `AnnualCrop` or `PermanentCrop`.
3. **Built Environment Misclassification (13.6%)**:
   - Boundary artifacts mimic high-contrast urban features (`Industrial`, `Residential`).
4. **Contextual & Linear Discontinuity (5.4%)**:
   - Severing of transport corridors (`Highway`).

*100 representative failures audited and cataloged in `failure_sample_audit.csv`.*

---

## Slide 8: Metric Validity Analysis
### Do PSNR, SSIM, and LPIPS Detect Downstream Failures?

| Metric | Pooled Spearman $\rho$ | Within-Bin (30–60%) $\rho$ | $\rho$ vs. Harmful Flips |
|---|---|---|---|
| **LPIPS (Perceptual)** | **-0.3639** | **-0.2784** | **+0.3730** |
| **SSIM (Full Image)** | +0.3634 | +0.2344 | -0.3665 |
| **PSNR (Full Image)** | +0.2017 | +0.1587 | -0.1994 |
| **SSIM (In-Mask)** | +0.1334 | +0.1442 | -0.1329 |
| **PSNR (In-Mask)** | **+0.1242** | **+0.1389** | **-0.1209** |

- **LPIPS and Full SSIM** are nearly tied and represent the best proxy metrics ($\rho \approx 0.36$).
- **In-Mask PSNR is unreliable** ($\rho = 0.12$). Fills that achieve mathematically low MSE inside the mask frequently trigger catastrophic classification errors.

---

## Slide 9: Recommendations & Limitations
1. **Practical Guidelines for Practitioners**:
   - **$< 30\%$ Clouds**: Inpainting (Nearest Neighbor or Telea) is practical and effective.
   - **$30–60\%$ Clouds**: If complete tiles are mandatory, use spatial Nearest Neighbor; however, whether mask-aware training beats fills must be read from the v2 paired comparison.
   - **$> 60\%$ Clouds**: **Do NOT inpaint**. Inpainting actively degrades classification and creates an 80% failure rate; tile dropping or mask-aware inference is essential.
2. **Study Limitations**:
   - Synthetic plain white clouds on 64x64 EuroSAT RGB do not model atmospheric cirrus or multi-spectral NIR/SWIR bands.
   - Prototype U-Net was time-boxed on CPU (2 epochs); results reflect this prototype setting.
