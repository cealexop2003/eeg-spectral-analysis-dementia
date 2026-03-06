# EEG SPECTRAL ANALYSIS FOR DEMENTIA CLASSIFICATION
## Project Summary & Results Log

**Date**: February-March 2026  
**Dataset**: Kaggle "EEG of Alzheimer's and Frontotemporal dementia"  
**Subjects**: 88 (29 CN, 36 AD, 23 FTD)  
**Sampling Rate**: 500 Hz

---

## PART 1: SPECTRAL LEAKAGE & WELCH ANALYSIS

### Synthetic Signal Analysis
- **Purpose**: Demonstrate spectral leakage effects
- **Signals**: 10.0 Hz (perfect periodicity) vs 10.33 Hz (imperfect)
- **Result**: Δf=0.1 Hz insufficient for 10.33 Hz (spectral leakage observed)

### Real EEG Analysis  
- **Channel**: Pz (parietal)
- **Segment**: 15s, eyes-closed resting state
- **Window Lengths**: 2s, 4s, 8s (50% overlap)
- **Window Types**: Rectangular, Hann, Hamming
- **Bands**: Theta (4-8 Hz), Alpha (8-12 Hz), Beta (13-30 Hz)

**Key Findings**:
- **4s window optimal**: Best trade-off between Δf (0.25 Hz) and K (~7 windows)
- **Hann window best**: Minimizes spectral leakage for non-periodic signals  
- **Group discrimination**: Clear separation CN vs AD/FTD in theta/alpha bands

---

## PART 2: FEATURE EXTRACTION & STATISTICS

### Simple Statistics (Feature Extraction)

**Preprocessing**:
- IIR Butterworth bandpass filter: 1-40 Hz, order 4
- Welch PSD: 4s window, 50% overlap, Hann window
- Channels: Pz + F3

**Features Extracted** (18 total, 9 per channel):
1. Relative band powers: delta, theta, alpha, beta
2. Power ratios: theta/alpha, delta/alpha, slowing_ratio
3. Peak Alpha Frequency (PAF)
4. Spectral Centroid (4-15 Hz)

**Key Observations**:
- **Theta/Alpha Ratio**: CN=1.82, AD=3.27, FTD=3.19 (Pz channel)
- **Alpha Power**: CN=10.4%, AD=6.6%, FTD=6.7% (strong suppression in dementia)
- **PAF**: CN=9.66 Hz, AD=9.19 Hz, FTD=9.49 Hz (alpha slowing)
- **Correlation**: Pz-F3 same features highly correlated (r=0.91 for theta/alpha)

---

### Advanced Statistics

**Assumptions Testing**:
- **Normality**: Only 1/18 features normal (F3_theta_rel)
- **Variance**: All features have equal variances (Levene test)
- **Recommended**: Kruskal-Wallis for 17/18 features, ANOVA for 1/18

**Univariate Tests** (before FDR):
- 7/18 features significant at p < 0.01
- Top feature: Pz_theta_alpha (p=0.0002, ε²=0.195)

**FDR Correction** (Benjamini-Hochberg):
- **7/18 features remain significant** after FDR correction
- All 7 features: p_FDR < 0.05

**Top Discriminative Features**:
1. Pz_theta_alpha (p_FDR=0.0037) ⭐
2. Pz_centroid_4_15 (p_FDR=0.0073)
3. Pz_alpha_rel (p_FDR=0.013)
4. Pz_slowing_ratio (p_FDR=0.013)
5. F3_theta_alpha (p_FDR=0.014)
6. Pz_delta_alpha (p_FDR=0.014)
7. F3_theta_rel (p_FDR=0.023)

**Post-hoc Tests** (Dunn/Tukey):
- **AD vs CN**: Highly significant for all top features (p < 0.001)
- **CN vs FTD**: Significant (p < 0.05)
- **AD vs FTD**: NOT significant (p > 0.05) ⚠️

**Effect Sizes**:
- Pz_theta_alpha: ε²=0.195 (large effect)
- Pz_centroid: ε²=0.163 (large effect)
- Others: ε²=0.10-0.14 (medium effects)

**PCA Analysis**:
- PC1: 51.2% variance, p=0.128 (NOT significant) ← PCA Paradox
- PC2: 16.7% variance, p=0.012 (significant)
- PC3: 12.1% variance, p=0.597 (NOT significant)
- **Interpretation**: Main variance = individual differences (noise), not group differences (signal)

---

### Classification Analysis (3-class: CN vs AD vs FTD)

**Models Tested**:
1. Logistic Regression (linear)
2. SVM with RBF kernel (non-linear)
3. Random Forest (ensemble)
4. K-Nearest Neighbors (instance-based)

**Validation**: Stratified 5-Fold Cross-Validation

**Results**:

| Model | Accuracy | Balanced Acc | Macro F1 |
|-------|----------|--------------|----------|
| **SVM (RBF)** ⭐ | **52.2%** ± 6.0% | **47.5%** ± 6.0% | **42.3%** ± 6.2% |
| Logistic Reg | 47.4% ± 14.5% | 45.6% ± 12.7% | 44.1% ± 12.2% |
| KNN | 48.7% ± 9.0% | 45.5% ± 9.3% | 41.9% ± 10.0% |
| Random Forest | 40.7% ± 8.9% | 38.5% ± 7.7% | 35.1% ± 10.2% |

**Baselines**:
- Random guess: 33.3% balanced accuracy
- Majority class: 40.9% accuracy
- **Improvement**: +42.5% over random (but still poor)

**SVM Confusion Matrix**:
```
           Pred: AD  CN  FTD
Actual AD:    26   9   1   (72% recall)
Actual CN:     9  19   1   (66% recall)
Actual FTD:   16   6   1   (4% recall!) ⚠️
```

**Per-Class Performance (SVM)**:
- AD: Precision=51%, Recall=72%, F1=60%
- CN: Precision=56%, Recall=66%, F1=60%  
- **FTD: Precision=33%, Recall=4%, F1=8%** ← CRITICAL FAILURE

**Feature Importance** (averaged across models):
- **F3 channel more important** (avg=0.163) than Pz (avg=0.141)
- Top features: F3_theta_rel, F3_alpha_rel, F3_centroid
- Paradox: Statistical tests favored Pz, but ML favors F3

---

## KEY INSIGHTS & CONCLUSIONS

### What Worked ✅
1. **CN vs Dementia separation**: Clear statistical differences
2. **Spectral features meaningful**: Theta/alpha ratio, PAF, centroid are valid biomarkers
3. **Welch method robust**: 4s window provides optimal SNR
4. **7 robust features**: Survive multiple comparison correction

### What Didn't Work ❌
1. **AD vs FTD discrimination**: No significant differences in spectral features
2. **3-class classification**: Only 47.5% balanced accuracy (too low for clinical use)
3. **FTD detection**: 4% recall is catastrophic
4. **Small sample size**: N=88 insufficient for robust ML (especially 23 FTD)
5. **Low signal-to-noise**: PCA shows group differences are secondary to individual variability

### Main Challenges
1. **PCA Paradox**: PC1 (51% variance) doesn't separate groups; PC2 (17%) does
2. **Class imbalance**: FTD underrepresented (26% of data)
3. **Limited features**: Only spectral features from 2 channels
4. **AD-FTD similarity**: Both show similar EEG slowing patterns

---

## RECOMMENDATIONS FOR IMPROVEMENT

1. **Binary Classification**: Test CN vs Dementia (AD+FTD merged)
2. **More channels**: Use all 19 channels for spatial information
3. **Connectivity features**: Coherence, phase synchronization
4. **Temporal features**: Entropy, complexity measures
5. **Ensemble approach**: Combine EEG with other biomarkers (MRI, cognitive tests)
6. **Larger dataset**: Collect more subjects, especially FTD
7. **Deep learning**: Try CNNs on raw EEG or spectrograms

---

## FILES GENERATED

**Part 1**:
- `results/part1/synthetic_results/` (spectral leakage plots)
- `results/part1/eeg_real_results/` (19 plots: PSD, band powers, window comparisons)

**Part 2 - Simple Stats**:
- `results/part2/simple_stats/features.csv` (88×20 matrix)
- Boxplots (Pz, F3), correlation heatmap, violin plots

**Part 2 - Advanced Stats**:
- `univariate_tests_all_features.csv`
- `top10_discriminative_features.csv`  
- `posthoc_all_features.csv`
- `pca_scores.csv`, `pca_group_tests.csv`
- Statistical plots (boxplots, PCA scatter, effect sizes)

**Part 2 - Classification**:
- `classification_results.csv`
- `classification_summary_report.txt`
- `feature_importance.csv`
- Confusion matrices, model comparison plots

---

**Next Experiment**: Binary classification (CN vs Dementia) to test hypothesis that 2-class problem is more tractable.
