# Revision notes (v2)
1. Fair comparison: every method is now scored by 3 classifiers (clean / mask_aware / fill_aware). `paired_comparison.csv` = masked+mask_aware vs fill+fill_aware, paired bootstrap CI.
2. Test-mask leakage removed: adapted classifiers train on `generate_shape_mask` (ellipses/rectangles); test uses `generate_cloud_mask`.
3. Flip reference is per classifier (clean-image prediction of the SAME classifier).
4. Taxonomy: fills only, thick clouds. Automatic part = prediction-shift table; visual labels are manual (`manual_label_sheet.csv`).
5. U-Net: full train set, 30 epochs, best-val checkpoint. evaluate_all now errors if any checkpoint is missing (no silent fallback).
6. Metric validity: clean classifier, fill methods only.
7. evaluate_all: UTF-8 stdout, traceback + explicit exit code.
## Report/slides TODO after rerun
- All numbers in PROJECT_REPORT.md / SLIDES.md are v1; replace from new CSVs.
- Remove "mask-aware is superior" claims unless `paired_comparison.csv` supports them.
- State: synthetic clouds, 64x64, held-out-mask training for adapted classifiers.
