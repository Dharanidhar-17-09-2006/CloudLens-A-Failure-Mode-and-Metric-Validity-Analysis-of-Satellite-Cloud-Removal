# Cloud Removal Failure-Mode Analysis: When Filling Helps Land-Cover Classification

**Course**: Visual Computing with AI/ML (AI61201)
**Team**: P. Sai Dharanidhar Ram (24AI10004), Ruthvik Devarakonda (24AI10023)
**Mentor**: Prof. Somdyuti Paul, Department of Artificial Intelligence, IIT Kharagpur

---

## 1. Question and short answer

Does filling cloud-covered pixels help a land-cover classifier, compared with leaving the cloud in, masking it, or training the classifier to handle the fill?

**Short answer (EuroSAT, synthetic thick clouds, full test split of 4,050 tiles, 3 seeds):**

- A classifier trained on **filled** tiles beats one trained on **masked** tiles at every coverage, in every seed. The gap is 5 points at 10–30%, 8 at 30–60% and 13 at 60–80%.
- The best fills are Telea and a small scratch U-Net, which are level. Nearest neighbour is about 1–3 points behind them at 30–80%. Mean fill is the weakest fill.
- Leaving the cloud in works only for a classifier that was never trained on fills. Once the classifier is trained on fills, it is far above cloudy as-is at every coverage.
- The pre-registered decision rule is met at all three coverage bins: fill, then train the classifier on filled tiles.

**Short answer (RICE, real clouds, no land-cover labels):** learned paired restorers give the best pixel fidelity on thick clouds. On thin clouds, no fill beats leaving the cloud in. Land-cover accuracy on real clouds was not measured; the classifier-agreement results are a proxy only.

---

## 2. Setup

- **Dataset**: EuroSAT, 27,000 Sentinel-2 RGB tiles, 64×64, 10 classes. Stratified splits (seed 42): train 18,900, validation 2,000, test 4,050 (full split).
- **Synthetic clouds**: noise-based cloud masks in three coverage bins (10–30%, 30–60%, 60–80%), opaque (thick) clouds. Test masks come from a different generator than the shape masks used for training, so test clouds are not seen in training.
- **Fills**: mean fill, nearest-neighbour (NN), Telea (OpenCV), and a small U-Net trained from scratch on EuroSAT. Dark-channel dehazing is not included in this comparison (see Section 6).
- **Classifiers**: ImageNet-initialised ResNet-18, full fine-tuning, 3 epochs, best-validation epoch kept. Three training regimes:
  - *clean*: trained on cloud-free tiles;
  - *mask-aware*: trained on cloud-free tiles with black holes from the shape-mask family;
  - *fill-aware*: trained on filled tiles, using the same four fill methods, with 20% cloud-free tiles.
- **Seeds**: three (0, 1, 2). Each seed retrains every classifier and regenerates the fill-aware training set.
- **Pre-registration**: the comparison, bins, statistics and decision rule were written in `ANALYSIS_PLAN.md` and committed before the full run.

---

## 3. Main result: land-cover accuracy on EuroSAT

Mean ± SD over three seeds. Cells are on the full test split (4,050 tiles per seed). Interval half-width per cell is about ±1.5 points (binomial, 95%).

**Fill-aware classifier (trained on filled tiles)**

| Input | 10–30% | 30–60% | 60–80% |
|---|---|---|---|
| Mean fill | 0.934 ± 0.009 | 0.885 ± 0.010 | 0.789 ± 0.008 |
| Nearest-neighbour fill | 0.943 ± 0.009 | 0.903 ± 0.012 | 0.799 ± 0.010 |
| Telea fill | **0.950 ± 0.007** | **0.916 ± 0.008** | **0.830 ± 0.015** |
| U-Net fill (scratch) | 0.948 ± 0.005 | 0.914 ± 0.006 | 0.826 ± 0.007 |
| Cloudy as-is (fill-aware classifier) | 0.682 ± 0.060 | 0.348 ± 0.103 | 0.151 ± 0.056 |

**Masked input, mask-aware classifier**

| Input | 10–30% | 30–60% | 60–80% |
|---|---|---|---|
| Masked (black fill) | 0.898 ± 0.014 | 0.833 ± 0.017 | 0.698 ± 0.022 |

**Clean classifier (trained on cloud-free tiles)**

| Input | 10–30% | 30–60% | 60–80% |
|---|---|---|---|
| Cloudy as-is | 0.730 ± 0.041 | 0.343 ± 0.032 | 0.137 ± 0.014 |
| Nearest-neighbour fill | 0.946 ± 0.009 | 0.810 ± 0.021 | 0.441 ± 0.039 |
| Telea fill | 0.942 ± 0.006 | 0.716 ± 0.032 | 0.267 ± 0.004 |
| U-Net fill (scratch) | 0.939 ± 0.007 | 0.689 ± 0.039 | 0.255 ± 0.019 |
| Mean fill | 0.912 ± 0.018 | 0.533 ± 0.074 | 0.144 ± 0.006 |

**Decision rule (pre-registered).** Fill-aware beats mask-aware in all three seeds at every bin. Recommendation: fill with a classical or learned method, and train the classifier on filled tiles.

**Findings**

1. **Fill-aware beats masked input everywhere.** Best fill minus masked input, per seed: +5.8, +4.4, +5.4 points at 10–30%; +9.4, +7.2, +8.3 at 30–60%; +14.6, +11.8, +14.2 at 60–80%.
2. **The earlier reversal is confirmed.** In the ablation, masked input looked best at heavy coverage only because the fills were scored by a classifier that had never seen fills. That result is superseded by this one.
3. **Telea and the scratch U-Net are level.** Differences are 0.2–0.4 points, within seed spread. Telea is nominally first in every bin, so state the tie, not a winner.
4. **The U-Net's standing depends on the classifier.** With a fine-tuned fill-aware classifier it matches Telea. With a frozen backbone (head-only, earlier ablation), it beats nearest neighbour by 5–6 points. Report both.
5. **Nearest neighbour is strong but not first.** It is 1–3 points behind Telea and the U-Net at 30–80%, and within 1 point at 10–30%.

**Interpretation.** With a small hole, the surrounding pixels say enough about what lies underneath that almost any reasonable fill recovers the class. With a large hole, the fill has to invent most of the texture, and a classifier that has learned how fills look can read that texture as a land class consistently. Training on fills is what makes the difference.

**Scope of the claim.** The fill-aware classifier was trained on outputs from the same four fill methods it is tested on. The result therefore means "adapting the classifier to the fill helps", not "any fill helps". Test tiles and test cloud masks were kept separate from training, so this is not leakage.

---

## 4. Opacity sweep: filling versus leaving the cloud in

EuroSAT, 30–60% coverage, 500 test tiles, clean classifier. This run used the clean classifier only, so it shows the fill-versus-cloud trade-off without the fill-aware effect.

| Opacity | Cloudy as-is | Telea | Mean fill |
|---|---|---|---|
| 0.20 | **0.826** | 0.584 | 0.478 |
| 0.50 | **0.650** | 0.584 | 0.478 |
| 0.65 | 0.592 | 0.584 | 0.478 |
| 0.80 | 0.520 | **0.584** | 0.478 |
| 1.00 | 0.408 | **0.584** | 0.478 |

Telea and mean fill are flat across opacity because they overwrite the masked pixels. With the clean classifier, leaving thin clouds in is better than filling up to about opacity 0.65. Thin clouds keep signal that fills discard, so the choice depends on both cloud opacity and classifier.

---

## 5. Real clouds: RICE (no land-cover labels)

Held-out tiles, 95% bootstrap intervals over images. LPIPS is lower-is-better. "Agreement" is the proportion of tiles where the classifier's prediction on the filled tile matches its prediction on the clear tile. It is a proxy, not accuracy.

**RICE1 (thin clouds, 626 tiles)**

| Method | LPIPS | Agreement |
|---|---|---|
| Cloudy as-is | **0.149** [0.135, 0.165] | 58.9% |
| Paired restorer (big) | 0.174 [0.149, 0.205] | 71.3% |
| Paired restorer (small) | 0.178 [0.154, 0.211] | 69.6% |
| Dark-channel prior (whole tile) | 0.189 [0.158, 0.231] | 69.6% |
| Mean fill | 0.355 [0.331, 0.382] | 55.1% |

**RICE2 (thick clouds, 717 tiles)**

| Method | LPIPS | Agreement |
|---|---|---|
| Cloudy as-is | 0.402 [0.364, 0.436] | 46.2% |
| EuroSAT-trained U-Net | 0.374 [0.355, 0.391] | 58.6% |
| RICE-trained U-Net (small) | 0.318 [0.295, 0.338] | 65.5% |
| Mean fill | 0.323 [0.299, 0.344] | 81.5% |
| Paired restorer (big) | 0.336 [0.309, 0.358] | 84.3% |

**Findings.** Training on real clouds helps compared with synthetic training (RICE2 LPIPS 0.374 to 0.318). Model size and the GAN add nothing. On thin clouds no fill beats leaving the cloud in; the paired restorer ties it, and intervals overlap. On thick clouds, paired restorers and mean fill tie on agreement. Mask dilation (3 px) lowers RICE2 fill LPIPS from about 0.32 to 0.24, and raises RICE1 fill LPIPS from about 0.35 to 0.50. The cause is not confirmed; estimated masks may miss a hazy rim.

---

## 6. Dehazing (supplementary, clean classifier only)

Dehazing was tested with the clean classifier and not in the fill-aware comparison. Results are reported as a supplementary arm.

- **EuroSAT:** the dark-channel prior (DCP), applied to the whole tile, is worse than leaving the cloud in at every opacity. Restricting the DCP output to the cloud mask improves it (0.758 against 0.584 at opacity 0.2) but it still does not beat leaving the cloud in (0.826).
- **RICE1 (thin, real):** DCP improves overall PSNR (18.6 against 16.8 dB, intervals do not overlap) and SSIM slightly. LPIPS is worse (0.189 against 0.149), and classifier agreement gains overlap.
- **RICE2 (thick):** DCP is worse than cloudy on every metric. This is the negative control.
- The learned paired restorer reaches 32 dB PSNR on RICE1 against 16.8 for cloudy, with non-overlapping intervals, but ties on LPIPS.

**Claim.** Dehazing improves pixel error on thin real clouds, but did not improve perceptual quality or land-cover classification in the tests run here. These tests used the clean classifier. Dehazing was not evaluated with a fill-aware classifier, so the result does not establish that dehazing is ineffective under that setting.

---

## 7. Metric validity and failure analysis

**Setup.** Metric validity was tested on the full EuroSAT test split (4,050 tiles, three coverage bins, four fill methods, 12 conditions), using the fill-aware classifier. A harmful flip is a tile the classifier labels correctly on the clear image and incorrectly on the filled image. Confidence intervals resample tiles, not individual measurements, because each tile contributes to all 12 conditions.

**Metric validity (Spearman correlation with harmful flips, pooled over all conditions, 95% tile-level bootstrap interval):**

| Metric | Expected sign | ρ | 95% CI |
|---|---|---|---|
| SSIM, whole tile | negative | −0.21 | (−0.22, −0.20) |
| LPIPS | positive | +0.22 | (+0.21, +0.23) |
| PSNR inside mask | negative | −0.12 | (−0.14, −0.11) |
| SSIM inside mask | negative | −0.09 | (−0.11, −0.08) |

Within each coverage bin, the correlations are weaker: whole-tile SSIM −0.08 to −0.12, LPIPS +0.07 to +0.12, PSNR inside the mask −0.06 to −0.14.

**Findings.**
1. Every metric moves in the expected direction, and every interval excludes zero. The associations are reliable, but none is strong: the largest explains about 4% of the variation in harmful flips.
2. Whole-tile SSIM and LPIPS are the most consistent across bins. Inside-mask PSNR is the strongest association at 60–80% coverage (ρ = −0.14), so a fill's in-mask error does carry some information about failure there.
3. No single metric is a dependable warning that a fill will mislead the classifier. A high score does not rule out a harmful fill.

**Harmful-flip rates by fill (%):**

| Fill | 10–30% | 30–60% | 60–80% |
|---|---|---|---|
| Telea | 1.5 | 4.9 | 13.9 |
| U-Net (scratch) | 1.7 | 5.4 | 14.1 |
| Nearest neighbour | 2.1 | 6.0 | 15.9 |
| Mean fill | 3.2 | 7.5 | 17.9 |

Telea has the lowest harmful rate in every bin, consistent with its top accuracy in Section 3. Mean fill has the highest in every bin.

**Failure types (indicative).** From 60 manually labelled failures in the earlier 200-tile run: nearest neighbour fails mainly through texture hallucination (10 of 15), Telea and the U-Net mainly through lost structure, and mean fill mainly through colour shift (13 of 15). The earlier 200-tile run also found that mean fill and Telea push harmful flips toward water classes (91% and 77% of their harmful flips predicted as SeaLake or River). These are hypotheses about mechanism, not tested causes.

## 8. Transfer learning

The classifiers use ImageNet-initialised ResNet-18 with full fine-tuning, and the frozen-backbone comparison from the earlier ablation is reported in Section 3. The U-Nets are trained from scratch. A satellite-pretrained encoder was not tested in this report.

---

## 9. Limitations

1. **Synthetic clouds on EuroSAT.** The clouds are opaque noise blobs, without cirrus, shadow or spectral effects.
2. **Fill-aware training uses the test fill methods.** The result shows that adapting the classifier to fills helps, not that any unseen fill would help equally.
3. **Classifier dependence.** Fill rankings change with the classifier (Section 3, finding 4).
4. **Single cloud generator for EuroSAT.** Results may shift with other cloud models.
5. **64×64 tiles.** Context is limited compared with full-scene evaluation.
6. **RICE has no land-cover labels.** Agreement is a proxy.
7. **RICE1 masks are estimated** by thresholding |cloudy − clear|, so RICE1 coverage bins are approximate.
8. **Dehazing was not evaluated with the fill-aware classifier** (Section 6).
9. **Failure-type labels are from an earlier 60-sample manual labelling** and are indicative only. Metric-validity correlations pool all fill methods; the within-method check (`within_method_validity.py`) is reported separately.
10. **Transfer learning with satellite pretraining was not tested.**

---

## 10. Conclusion

When a small area is missing, filling it and classifying the result is almost as good as using the real pixels, and any reasonable fill works. When a large area is missing, the best strategy is to fill and train the classifier on filled tiles. Simple fills are as good as learned ones once the classifier is trained on fills. The main limitation is that the classifier learns from the same fill methods it is tested on, so the claim is about adapting the classifier, not about every fill.
