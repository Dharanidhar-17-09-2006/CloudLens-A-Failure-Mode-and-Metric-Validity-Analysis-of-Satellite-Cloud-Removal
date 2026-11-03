# EuroSAT Dataset Audit Notes

## 1. Overview & Dataset Profile
- **Dataset**: EuroSAT (Sentinel-2 satellite imagery for land-use and land-cover classification)
- **Source**: Helber et al. (2019), "EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification"
- **Total Samples**: 27000 images
- **Image Size**: 64 x 64 pixels, 3 channels (RGB)
- **Spatial Resolution**: 10 meters per pixel (European Sentinel-2 satellite)
- **License**: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) / Open Access for academic research.

## 2. Land-Cover Classes (10 categories)
| Class ID | Class Name | Total Count | Train (70%) | Val (15%) | Test (15%) |
|---|---|---|---|---|---|
| 0 | AnnualCrop | 3000 | 2100 | 450 | 450 |
| 1 | Forest | 3000 | 2100 | 450 | 450 |
| 2 | HerbaceousVegetation | 3000 | 2100 | 450 | 450 |
| 3 | Highway | 2500 | 1750 | 375 | 375 |
| 4 | Industrial | 2500 | 1750 | 375 | 375 |
| 5 | Pasture | 2000 | 1400 | 300 | 300 |
| 6 | PermanentCrop | 2500 | 1750 | 375 | 375 |
| 7 | Residential | 3000 | 2100 | 450 | 450 |
| 8 | River | 2500 | 1750 | 375 | 375 |
| 9 | SeaLake | 3000 | 2100 | 450 | 450 |

## 3. Image Integrity & Uniformity
- **Resolution**: Uniform 64 x 64 pixels across all sampled images.
- **Color Mode**: Standard RGB (8-bit per channel, 0-255).
- **Split Strategy**: Stratified random split with fixed seed (`seed=42`) ensuring identical class balance across train, validation, and test partitions.
- **Downstream Utility**: Standard benchmark for land use and land cover (LULC) classification.

## 4. Relevance to Cloud Inpainting Scope
- Complete tiles are required by standard CNN architectures (e.g., ResNet-18).
- Clean images serve as ground truth for synthetic cloud corruption, inpainting restoration, and downstream accuracy comparison across cloud opacity and coverage bins (10-30%, 30-60%, 60-80%).
