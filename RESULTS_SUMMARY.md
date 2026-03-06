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

## BINARY CLASSIFICATION (CN vs Dementia)

### Baseline Binary Classification (2-channel: Pz, F3)

**Objective**: Test if merging AD+FTD improves performance

**Setup**:
- Binary labels: CN=0, Dementia (AD+FTD)=1
- Class distribution: 29 CN, 59 Dementia
- Features: Same 18 features from Pz + F3
- Validation: Stratified 5-Fold CV

**Results**:

| Model | Balanced Acc | F1-Score | ROC AUC | Sensitivity | Specificity |
|-------|--------------|----------|---------|-------------|-------------|
| **Logistic Reg** ⭐ | **69.5%** | **79.5%** | **77.5%** | **83.1%** | **55.2%** |
| SVM (RBF) | 66.7% | 76.7% | 69.3% | 74.6% | 58.6% |
| Random Forest | 64.4% | 75.5% | 72.4% | 76.3% | 51.7% |
| KNN | 62.6% | 76.1% | 65.9% | 79.7% | 44.8% |

**Confusion Matrix (Logistic Regression)**:
```
           Pred: CN  Dementia
Actual CN:     16      13      (55.2% specificity)
Dementia:      10      49      (83.1% sensitivity)
```

**Key Findings**:
- **+46% improvement** over 3-class (69.5% vs 47.5%)
- Good sensitivity (83.1%) for dementia detection
- Moderate specificity (55.2%) - some false positives
- **Clinical utility**: Reasonable screening performance

---

### Optimized Binary Classification

**Objective**: Maximize performance through hyperparameter tuning

**Strategy**:
1. Feature selection comparison (7 feature sets)
2. Hyperparameter tuning (GridSearchCV)
3. Class weighting (balanced)
4. Ensemble voting

**Feature Selection Results**:

| Feature Set | Best Balanced Acc |
|-------------|------------------|
| **All 18 features** ⭐ | **69.5%** |
| Top 7 (FDR-significant) | 63.0% |
| Top 10 | 67.0% |
| Pz only (9 features) | 66.5% |
| F3 only (9 features) | 65.0% |
| Power ratios only | 62.5% |
| Simple powers only | 59.0% |

**Finding**: All 18 features outperform feature selection

**Hyperparameter Tuning Results**:

| Model | Best Params | Balanced Acc | Improvement |
|-------|-------------|--------------|-------------|
| **Logistic Reg** ⭐ | C=0.01, balanced weight | **74.5%** | **+7.2%** |
| SVM (RBF) | C=1, γ=0.01, balanced | 73.6% | +10.3% |
| Random Forest | depth=3, balanced_subsample | 68.3% | +6.0% |

**Final Best Model (Logistic Regression)**:
- Balanced Accuracy: **74.5%** ± 8.2%
- F1-Score: 80.1%
- ROC AUC: 75.0%
- Sensitivity: 76.3%
- Specificity: 72.4%

**Feature Importance**:
Top 3 features: Pz_theta_alpha, Pz_centroid_4_15, F3_theta_alpha

---

## 4-CHANNEL EXPANSION EXPERIMENT

### Feature Extraction (4-channel: Pz, F3, F4, O1)

**Objective**: Test if additional channels improve performance

**Added Channels**:
- **F4** (frontal-right): For hemispheric asymmetry (F3 vs F4)
- **O1** (occipital): For posterior alpha rhythm enhancement

**Features Extracted**: 36 total (9 per channel × 4 channels)

**Hypothesis**: 
- F3 vs F4 asymmetry may differentiate FTD (frontal pathology)
- O1 alpha may differentiate AD (posterior involvement)

---

### Binary Classification: 2-channel vs 4-channel

**Results**:

| Model | 2-ch Bal Acc | 4-ch Bal Acc | Change |
|-------|--------------|--------------|---------|
| **Logistic Reg** | **74.5%** | 71.6% | **-3.9%** ⬇️ |
| SVM (RBF) | 73.6% | 73.3% | -0.5% ⬇️ |
| Random Forest | 68.3% | 69.9% | +2.4% ⬆️ |

**Best Overall**: Logistic Regression with 2-ch (Pz, F3) = **74.5%**

**Key Findings**:
- ❌ **4 channels WORSENED binary classification** (avg -1.3%)
- Curse of dimensionality: 36 features ÷ 88 samples = 0.41 ratio (overfitting)
- F4 and O1 added noise, not discriminative signal
- **Pz + F3 is optimal 2-channel selection** for spectral features

---

### 3-Class Classification: 2-channel vs 4-channel

**Results**:

| Model | 2-ch Bal Acc | 4-ch Bal Acc | Change | FTD Recall (2-ch) | FTD Recall (4-ch) |
|-------|--------------|--------------|---------|-------------------|-------------------|
| Logistic Reg | 42.5% | 38.9% | **-8.5%** | 39.1% | 30.4% ⬇️ |
| **SVM (RBF)** | 44.2% | **46.1%** | **+4.3%** ⭐ | 21.7% | 26.1% ⬆️ |
| Random Forest | 38.0% | 44.1% | +16.1% | 8.7% | 17.4% ⬆️ |

**Best Overall**: SVM (RBF) with 4-ch = **46.1%** (marginal improvement)

**Key Findings**:
- ⚠️ **Minimal improvement** for 3-class (+4.3% for SVM)
- ❌ **FTD recall still catastrophic** (26.1% < 33% random)
- ❌ Logistic Regression WORSENED with 4 channels
- **Overall: 4 channels DON'T solve AD vs FTD discrimination**

---

## UPDATED CONCLUSIONS

### Performance Summary

| Task | Best Model | Best Accuracy | Status |
|------|-----------|---------------|---------|
| **Binary (CN vs Dementia)** | Logistic Reg (2-ch) | **74.5%** | ✅ Reasonable |
| **3-Class (CN/AD/FTD)** | SVM (4-ch) | **46.1%** | ❌ Poor |

### What Worked ✅
1. **Binary classification**: 74.5% is clinically reasonable for spectral-only features
2. **Hyperparameter tuning**: +7.2% improvement (69.5% → 74.5%)
3. **Class weighting**: Critical for imbalanced datasets (29 CN vs 59 Dementia)
4. **2-channel selection (Pz + F3)**: Optimal for spectral features
5. **All 18 features**: Outperformed feature selection (69.5% vs 63%)

### What Didn't Work ❌
1. **4-channel expansion**: Worsened binary (-3.9%), minimal 3-class improvement (+4.3%)
2. **AD vs FTD discrimination**: Remains unsolved (post-hoc p=0.51)
3. **FTD detection**: 26.1% recall (worse than random)
4. **Feature selection**: Top 7 FDR-significant underperformed all 18
5. **Curse of dimensionality**: 36 features too many for N=88

### Why 4-Channels Failed
1. **No new discriminative information**: F4, O1 spectral patterns similar to F3, Pz
2. **Overfitting**: 36 features ÷ 88 samples = high feature-to-sample ratio
3. **Added noise**: More features dilute signal without adding value
4. **Pz + F3 already optimal**: Cover parietal slowing (Pz) + frontal dysfunction (F3)

### Performance Barriers (Why can't reach 90%)
1. **Spectral features limited**: Power-based features can't distinguish AD vs FTD
2. **Small sample size**: N=88 insufficient (need N≥200)
3. **Low SNR**: PCA shows group signal < individual noise (PC1 51% vs PC2 17%)
4. **Homogeneous dementia EEG**: Both AD and FTD show theta/alpha slowing

---

## REALISTIC NEXT STEPS (Without Deep Learning or More Data)

### Option 1: Connectivity Features ⭐ RECOMMENDED
**What**: Coherence / Phase Locking Value (PLV) between channel pairs

**Why**: Captures "communication" between brain regions
- **AD**: Long-range connectivity disruption (F3↔Pz coherence ↓↓)
- **FTD**: Frontal local connectivity disruption (F3↔F4 coherence ↓↓)

**Implementation**:
- 4 channels → 6 pairs (F3-F4, F3-Pz, F4-Pz, F3-O1, F4-O1, Pz-O1)
- 3 bands (theta, alpha, beta) → 18 connectivity features
- Total: 36 spectral + 18 connectivity = **54 features**

**Expected Improvement**:
- Binary: 74.5% → **78-82%** (realistic)
- 3-class: 46.1% → **50-55%** (realistic)

### Option 2: Complexity Features
- Sample entropy, permutation entropy
- Fractal dimension, Hurst exponent
- 4-8 additional features

### Option 3: Advanced Spectral
- Individual alpha peak analysis
- Theta/beta ratio
- Spectral asymmetry indices

---

## FILES GENERATED (Complete List)

**Part 1**:
- `results/part1/synthetic_results/` (spectral leakage plots)
- `results/part1/eeg_real_results/` (19 plots)

**Part 2 - Simple Stats**:
- `results/part2/simple_stats/features.csv` (88×20, 2-channel)
- `results/part2/simple_stats_4ch/features.csv` (88×38, 4-channel)
- Boxplots, correlation heatmaps, violin plots

**Part 2 - Advanced Stats**:
- Statistical test results (CSVs)
- PCA analysis
- Effect size plots

**Part 2 - Classification**:
- `results/part2/classification_results.csv` (3-class, 2-ch)
- `results/part2/binary_classification/` (binary, 2-ch baseline)
- `results/part2/optimized_binary/` (binary, 2-ch tuned)
- `results/part2/4channel_comparison/` (2-ch vs 4-ch, binary + 3-class)

---

**Current Status**: 74.5% binary classification with 2-channel spectral features.  
**Next Experiment**: Connectivity features (coherence/PLV) to improve both binary and 3-class performance.
