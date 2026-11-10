> **DRAFT v2. Items marked [PENDING] need the run listed in the "Pending runs" section. Do not copy v1 tables from the old report.**

# Cloud Removal Failure-Mode Analysis: Empirical Report (v2 draft)

**Course**: Visual Computing with AI/ML (AI61201)
**Team**: P. Sai Dharanidhar Ram (24AI10004), Ruthvik Devarakonda (24AI10023)
**Mentor**: Prof. Somdyuti Paul, Department of Artificial Intelligence, IIT Kharagpur

---

## 1. Question and short answer

Does filling cloud-covered pixels help a land-cover classifier, compared with leaving the cloud in, masking it, or adapting the classifier to missing pixels?

**Short answer (EuroSAT, synthetic thick clouds, 3 seeds, fixed classifier configurations):**
- **Small clouds (10-30%)**: filling beats leaving the cloud in by about 17-20 points (0.75 to 0.93-0.94 for nearest-neighbour, Telea and U-Net fills), and beats the masked-input classifier by 3-4 points (0.906). This holds in all three seeds.
- **Medium clouds (30-60%)**: a nearest-neighbour fill (0.812) comes close to the masked-input classifier (0.848), which is ahead in 2 of 3 seeds. Telea (0.715) and the U-Net (0.690) trail it by 13-16 points.
- **Heavy clouds (60-80%)**: the masked-input classifier (0.712) beats the best fill (nearest neighbour, 0.466) by about 25 points, in every seed.
- **A learned fill (U-Net) did not beat a nearest-neighbour fill** when the classifier is fine-tuned. With a frozen backbone (head-only) the U-Net is ahead of nearest neighbour by 5-6 points at 30-80%.

**Short answer (RICE, real clouds, no land-cover labels):** learned paired restorers give the best pixel fidelity and tie the best simple fill on classifier agreement for thick clouds. On thin clouds they tie leaving the cloud in. Land-cover accuracy on RICE was not measured; the proxy is agreement with the classifier's clear-tile prediction.

---

## 2. Setup

- **Dataset**: EuroSAT (27,000 Sentinel-2 RGB tiles, 64×64, 10 classes). Stratified split, seed 42: train 18,900, validation 4,050, test 4,050. The ablation uses 2,000 random test tiles per bin, sampled with a fixed seed.
- **Synthetic clouds**: multi-scale noise thresholded to a target coverage bin (10–30%, 30–60%, 60–80%), opaque (`thick`). Opacity sweep uses alpha 0.2 to 1.0 on 30–60% coverage.
- **Classifiers** (all ImageNet-initialised ResNet-18, 10 classes):
  - *Head only*: train `fc`.
  - *Layer4*: train `layer4` and `fc`.
  - *Full*: train all layers (learning rate 3e-4; the others use 1e-3).
  - Each trained two ways: **clean** (no clouds) or **mask-aware** (80% of training tiles get random shape masks, filled with black). The shape masks come from a family separate from the test cloud family.
- **Input conditions at test time**: `clean`, `cloudy` (raw), `masked_zero` (black fill), `mean_fill`, `telea` (OpenCV Telea).
- **Metrics**: accuracy per coverage bin. 95% intervals are ±1.96·√(p(1−p)/n); at n = 2,000 they are about ±2 points.
- **Runs**: three seeds (0, 1, 2), 3 epochs each, best-validation epoch kept. Validation for masked-trained models uses masked tiles (30-60%), for clean-trained models clean tiles.

---

## 3. Main result: land-cover accuracy on EuroSAT (3 seeds)

Mean ± sample SD over 3 seeds (0, 1, 2). Epochs are chosen on a validation set (best-validation epoch). Configurations are **fixed in advance, not chosen on the test set**: fill rows and the cloudy row use the full-fine-tuned classifier trained on clean tiles (full/clean); the masked-input row uses the full-fine-tuned classifier trained on masked tiles (full/mask-aware). Clean test accuracy: full/clean 0.952-0.966, full/mask-aware 0.936-0.950.

| Input (what the classifier sees) | 10-30% | 30-60% | 60-80% |
|---|---|---|---|
| Cloudy as-is (full/clean) | 0.749 ± 0.015 | 0.443 ± 0.049 | 0.206 ± 0.050 |
| Masked (black fill), mask-aware classifier | 0.906 ± 0.004 | **0.848 ± 0.016** | **0.712 ± 0.015** |
| Mean fill, then classify (full/clean) | 0.912 ± 0.017 | 0.541 ± 0.054 | 0.144 ± 0.022 |
| Nearest-neighbour fill (full/clean) | **0.944 ± 0.010** | 0.812 ± 0.036 | 0.466 ± 0.037 |
| Telea fill (full/clean) | 0.938 ± 0.014 | 0.715 ± 0.034 | 0.265 ± 0.020 |
| U-Net fill, EuroSAT-trained (full/clean) | 0.934 ± 0.015 | 0.690 ± 0.024 | 0.269 ± 0.030 |

**Frozen-backbone classifier (head-only, trained on clean tiles), same fills:**

| Fill | 10-30% | 30-60% | 60-80% |
|---|---|---|---|
| U-Net | 0.860 ± 0.009 | 0.696 ± 0.023 | 0.455 ± 0.034 |
| Nearest neighbour | 0.852 ± 0.007 | 0.646 ± 0.012 | 0.398 ± 0.009 |
| Telea | 0.853 ± 0.008 | 0.652 ± 0.039 | 0.318 ± 0.039 |

The U-Net minus nearest-neighbour difference is positive in all 3 seeds for head-only (+0.8, +5.1, +5.7 points) and negative in all 3 seeds for full/clean (-1.0, -12.2, -19.7 points). The ranking of fills therefore depends on the classifier. A possible reason (not tested) is that a frozen ImageNet backbone copes better with smooth fills, and a fine-tuned one does better with sharp nearest-neighbour edges.

**Findings**
1. **Small clouds (10-30%)**: nearest-neighbour, Telea and U-Net fills (0.93-0.94) beat the masked-input classifier (0.906) by 3-4 points in every seed, and beat leaving the cloud in (0.749) by about 19 points. Mean fill (0.912) ties the masked-input classifier.
2. **Medium clouds (30-60%)**: the masked-input classifier (0.848) is ahead of nearest neighbour (0.812) by 3.6 points, inside the seed spread (one seed reverses it). Telea and the U-Net trail by 13-16 points.
3. **Heavy clouds (60-80%)**: the masked-input classifier (0.712) beats the best fill (nearest neighbour, 0.466) by 25 points in every seed. The learned U-Net (0.269) is no better than Telea.
4. **Learned versus simple fill**: with a fine-tuned classifier the U-Net is level with Telea and below nearest neighbour; with a frozen backbone it leads.

**Interpretation**: with a small hole, the surrounding pixels say enough about what is underneath that any reasonable fill recovers the class. With a large hole, a fill mostly supplies a guessed texture, and wrong guesses look like real land to the classifier. A classifier that is told which pixels are missing does not have to trust them.

**Caveat on selection**: the full six-configuration grid is in the appendix. Choosing the best configuration per cell on the test set gives similar but slightly higher numbers and is not used for any claim above.

---

## 4. Opacity sweep: filling against leaving the cloud in, and dehazing

EuroSAT, 30-60% coverage, 500 test tiles, clean classifier (`resnet18_clean.pth`). Accuracy:

| Opacity | Cloudy as-is | Telea | Mean fill | Dark-channel prior | Dark-channel prior, mask only |
|---|---|---|---|---|---|
| 0.20 | **0.826** | 0.584 | 0.478 | 0.584 | 0.758 |
| 0.35 | **0.712** | 0.586 | 0.478 | 0.496 | 0.664 |
| 0.50 | **0.650** | 0.584 | 0.478 | 0.406 | 0.558 |
| 0.65 | **0.592** | 0.584 | 0.478 | 0.346 | 0.480 |
| 0.80 | 0.520 | **0.584** | 0.478 | 0.302 | 0.390 |
| 0.95 | 0.452 | **0.584** | 0.478 | 0.246 | 0.310 |
| 1.00 | 0.408 | **0.584** | 0.478 | 0.226 | 0.294 |

LPIPS (lower is better): Telea 0.179 and mean fill 0.190 at every opacity. Cloudy as-is: 0.132 (0.2), 0.234 (0.35), 0.322 (0.5), 0.395 (0.65), 0.454 (0.8), 0.505 (1.0). Dark-channel prior, mask only: 0.163, 0.231, 0.310, 0.395, 0.493, 0.606.

**Findings**
- Leaving the cloud in is better than filling up to about opacity 0.65 on accuracy; a fill is better from opacity 0.8 up.
- Mean fill and Telea are flat across opacity, because they overwrite the masked pixels and discard the cloud's own signal.
- **Dehazing did not help classification at any opacity.** The dark-channel prior restricted to the mask is better than the whole-tile version (0.758 against 0.584 at opacity 0.2), but still below leaving the cloud in. Its LPIPS is marginally below cloudy at opacity 0.35-0.5 (0.231 against 0.234, 0.310 against 0.322); these differences are small and no interval was computed, so no gain is claimed.
- Nearest-neighbour fill was not included in the sweep.

---

## 5. Real clouds: RICE (no land-cover labels)

Held-out tiles, 95% bootstrap intervals over images. The metric is LPIPS (lower is better), plus agreement with the classifier's clear-tile prediction (a proxy for land cover, not a label).

**RICE1 (thin clouds, 626 tiles)**

| Method | LPIPS | Agreement with clear tile |
|---|---|---|
| Cloudy as-is | **0.149** [0.135, 0.165] | 58.9% |
| Paired restorer (big) | 0.174 [0.149, 0.205] | 71.3% |
| Dark-channel prior | 0.189 [0.158, 0.231] | 69.6% |
| Paired restorer (small) | 0.178 [0.154, 0.211] | 69.6% |
| Mean fill | 0.355 [0.331, 0.382] | 55.1% |
| RICE U-Net (big + GAN) | 0.298 [0.274, 0.324] | 47.0% |

**RICE2 (thick clouds, 717 tiles)**

| Method | LPIPS | Agreement with clear tile |
|---|---|---|
| Cloudy as-is | 0.402 [0.364, 0.436] | 46.2% |
| EuroSAT U-Net | 0.374 [0.355, 0.391] | 58.6% |
| Nearest neighbour | 0.341 [0.313, 0.367] | 56.6% |
| RICE U-Net (small) | 0.318 [0.295, 0.338] | 65.5% |
| Mean fill | **0.323** [0.299, 0.344] | 81.5% |
| Paired restorer (big) | 0.336 [0.309, 0.358] | **84.3%** |

**Findings**
1. Training on real clouds helps against training on synthetic ones (RICE2 LPIPS 0.374 → 0.318).
2. Model size and the GAN add nothing: 1.9M, 17M and GAN give about the same LPIPS.
3. On thin clouds, no fill beats leaving the cloud in on LPIPS. Paired restorers tie it. Intervals overlap, so this is a tie, not a win.
4. On thick clouds, agreement is highest for paired restorers and mean fill, which are tied.
5. Mask dilation (3 px) improves RICE2 fills (LPIPS about 0.32 → 0.24) and worsens RICE1 fills (about 0.35 → 0.50). The reason is not confirmed; a likely cause is that estimated masks miss a hazy rim. [Unverified.]

---

## 6. Metric validity

- **In-mask PSNR and LPIPS disagree.** On RICE2, mean fill has the best PSNR (26.4 dB) and agreement (81.5%), but not the best LPIPS. A metric that rewards flat colour can score well on a fill that looks wrong.
- On RICE1, the dark-channel prior beats cloudy as-is on PSNR (18.6 vs 16.8 dB) and SSIM (0.79 vs 0.77), but loses on LPIPS (0.189 vs 0.149).
- On EuroSAT, mean fill and Telea are flat across opacity on every metric, so metrics cannot see the difference between thin and thick clouds through the fill.
- [PENDING: rank correlation of LPIPS, SSIM and PSNR with the classifier outcome, rerun on v2 outputs. The v1 correlation table is not valid for this draft.]

---

## 7. Failure analysis

- [PENDING: harmful and helpful flip counts from v2 `audit_failures.py`.]
- The v1 taxonomy (water-body collapse, vegetation smear, built-environment errors) came from the old protocol and has not been checked on v2. Treat it as a hypothesis until the v2 flips are counted.
- The manual failure labels were completed earlier, on the old protocol. [Check before reporting: were they produced on the v1 or v2 fills?]

---

## 8. Limitations

1. **Synthetic clouds on EuroSAT.** Clouds are opaque noise blobs. They do not model cirrus, shadows or spectral response.
2. **Three seeds, 3 epochs.** An earlier run that kept the last epoch showed an instability in one seed (full/clean validation 0.889 at epoch 3); the final runs keep the best-validation epoch and show no such dip. Differences of about 2 points or less are within noise.
3. **Configuration choice.** The headline table uses configurations fixed in advance. The fill ranking changes with the classifier (Section 3), so the claim is stated per classifier.
4. **Fill choice**: Telea was the classical fill used. [Confirm it was named before the results.]
5. **Dehazing**: a single classical dehazer (dark-channel prior, whole tile and mask-only) was tested. Other dehazers were not.
6. **RICE has no land-cover labels.** The agreement metric is a proxy. The fill-aware classifier may favour flat tiles (unchecked).
7. **RICE1 masks are estimated** by thresholding |cloudy − clear|, so the RICE1 coverage bins are approximate.
8. **64×64 tiles.** Context is limited compared with published full-tile results.
9. **Transfer**: the COCO-trained comparison from the original plan was dropped. EuroSAT-to-RICE transfer replaces it.
10. **Dehazing is supplementary.** The main study is thick clouds. Dehazing and inpainting are different problems, as Prof. Paul noted.

---

## 9. Pending runs before submission

| Run | Command | Fills which section |
|---|---|---|
| Full v2 EuroSAT evaluation | `python cloud_removal_project/src/evaluate_all.py` | 3 (nearest neighbour, U-Net rows), 6, 7 |
| Failure flips on v2 | `python cloud_removal_project/src/audit_failures.py` | 7 |
| Figures | `python cloud_removal_project/src/generate_figures.py` | all |
