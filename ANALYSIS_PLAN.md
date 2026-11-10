# Pre-registered analysis plan

Commit this file to git before running any comparison below. Results that disagree with this plan are still reported.

## Question
Does filling cloud-covered pixels help a land-cover classifier, compared with (a) leaving the cloud in, (b) masking it and using a classifier trained on masked tiles, and (c) filling and then using a classifier trained on filled tiles?

## Datasets
- **EuroSAT (controlled, labelled, synthetic thick clouds).** Full test split, 4,050 tiles. Splits from `data/splits/`, seed 42.
- **SEN12MS-CR subset (real clouds, labelled).** Streamed from `Hermanni/sen12mscr`, RGB bands (B4, B3, B2). Labels: IGBP land cover from SEN12MS, one label per tile = dominant class, tiles with dominant-class share below 0.7 are dropped. Per-season quota. Test keys are excluded from any pretraining.

## Comparisons (all three, every seed)
1. **Masked input**: black-filled tiles, classifier trained on masked tiles (`mask_aware`).
2. **Fill then clean classifier**: nearest neighbour, Telea, mean fill and the scratch U-Net, scored by a classifier trained on clean tiles (`clean`).
3. **Fill then fill-aware classifier**: the same four fills, scored by a classifier trained on fills from those same four methods, 20% clean tiles (`fill_aware`).

Classifier: ImageNet ResNet-18, full fine-tuning, best-validation epoch. Validation for `mask_aware` uses masked validation tiles; for `clean` and `fill_aware`, clean and filled validation tiles respectively.

## Coverage bins
10–30%, 30–60%, 60–80%, thick (opaque) clouds, as in the ablation.

## Seeds
Three seeds (0, 1, 2). Each seed retrains every classifier. The fill-aware training set is regenerated per seed.

## Statistics
- Accuracy per cell, 95% binomial interval on the full test split.
- Paired bootstrap over tiles for each difference reported in the text.
- Mean and sample SD over seeds for the headline table.
- No threshold or configuration is chosen on the test split.

## Decision rule (stated before results)
- If `fill_aware` beats `mask_aware` at a bin in all three seeds, the recommendation for that bin is: fill with the best fill and train the classifier on filled tiles.
- If `mask_aware` beats every fill option in all three seeds, the recommendation is: mask the cloud and train on masked tiles.
- Otherwise, the report says the result is seed- or method-dependent and does not give a recommendation for that bin.

## Transfer-learning check (separate, reported as separate columns)
- U-Net, random-init ResNet encoder
- U-Net, ImageNet-init ResNet encoder
- U-Net, MAE-pretrained encoder on SEN12MS-CR cloud-free training tiles
- The scratch EuroSAT U-Net is kept as a labelled reference, not part of the comparison.

## Limits (to be stated in the report)
- Synthetic thick clouds on EuroSAT.
- SEN12MS-CR labels are MODIS-derived (500 m), so they are coarse.
- RICE has no land-cover labels; its results are a proxy only.
- The recommendation depends on the classifier used.

## Amendment 1: SEN12MS-CR tile construction (written before any SEN12MS-CR run)
- RGB bands B4, B3, B2; reflectance / 3000 clipped to [0, 1], stored as 8-bit.
- Cloud mask = estimate: cloudy RGB-mean brighter than the cloud-free RGB-mean by more than 0.05 reflectance, 3x3 opening. Cloudy and clear images are different dates, so surface change can enter the mask and cloud shadows are missed.
- Tiles: four non-overlapping 64x64 crops per 256x256 patch, native 10 m.
- Labels: simplified IGBP (10 classes). One label per tile = dominant class; the tile is kept only if that class covers at least 70% of the tile and at least 90% of its pixels are valid.
- Split by (season, scene) hash, 70/10/20, so neighbouring patches never straddle splits. Test tiles are excluded from classifier training, fill-aware training and any encoder pretraining.
- Classes differ from EuroSAT, so classifiers for this test are trained on SEN12MS-CR train tiles (ImageNet ResNet-18, same recipe as the EuroSAT runs). EuroSAT classifiers are not reused.
- Test tiles are binned by estimated coverage (10-30, 30-60, 60-80%). SEN12MS-CR clouds are a mix of thin and thick, so results are not directly comparable to the thick-cloud EuroSAT results.
