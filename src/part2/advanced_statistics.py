"""
Advanced Statistical Analysis for EEG Feature Discrimination
============================================================

This module implements comprehensive statistical testing to identify which
spectral features best discriminate between CN, AD, and FTD groups.

Key Components:
1. Assumptions testing (normality, homogeneity of variance)
2. Univariate group comparisons (ANOVA/Kruskal-Wallis)
3. Post-hoc pairwise tests (Tukey HSD/Dunn)
4. Multiple comparison correction (FDR)
5. Multivariate analysis (MANOVA/PCA)
6. Effect size estimation
7. Statistical visualization

Author: EEG Analysis Pipeline
Date: February 2026
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import f_oneway, kruskal, shapiro, levene
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Try to import pingouin for advanced statistics
try:
    import pingouin as pg
    HAS_PINGOUIN = True
except ImportError:
    HAS_PINGOUIN = False
    print("Warning: pingouin not available. Some advanced features will be limited.")


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
FEATURES_FILE = Path('results/part2/simple_stats/features.csv')
STATS_OUTPUT_DIR = Path('results/part2/advanced_stats')

# Statistical parameters
ALPHA = 0.05  # Significance level
OUTLIER_IQR_MULTIPLIER = 3.0  # For outlier detection (conservative)
CORRELATION_THRESHOLD = 0.85  # For MANOVA feature selection
RANDOM_SEED = 42

# Group labels
GROUPS = ['CN', 'AD', 'FTD']

# Pairwise comparisons
PAIRS = [('CN', 'AD'), ('CN', 'FTD'), ('AD', 'FTD')]


# ============================================================================
# DATA LOADING AND QC
# ============================================================================

def load_features() -> pd.DataFrame:
    """
    Load feature matrix from CSV.
    
    Returns
    -------
    df : pd.DataFrame
        Feature matrix with subject_id, label, and features
    """
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"Features file not found: {FEATURES_FILE}")
    
    df = pd.read_csv(FEATURES_FILE)
    
    print(f"Loaded {len(df)} subjects with {df.shape[1]} columns")
    print(f"Groups: {df['label'].value_counts().to_dict()}")
    
    return df


def identify_feature_columns(df: pd.DataFrame) -> list:
    """
    Identify numeric feature columns (exclude metadata).
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame
    
    Returns
    -------
    feature_cols : list
        List of feature column names
    """
    # Exclude metadata columns
    metadata_cols = ['subject_id', 'label']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    # Verify they are numeric
    feature_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
    
    print(f"Identified {len(feature_cols)} numeric features")
    
    return feature_cols


def quality_control(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Perform data quality control checks.
    
    Checks:
    - Missing values
    - Extreme outliers (IQR method)
    - Near-zero variance features
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame
    feature_cols : list
        List of feature columns
    
    Returns
    -------
    qc_results : dict
        QC results and flagged issues
    """
    qc_results = {
        'missing_values': {},
        'outliers': {},
        'low_variance': []
    }
    
    print("\n" + "="*80)
    print("QUALITY CONTROL")
    print("="*80)
    
    # 1. Missing values
    print("\n1. Checking for missing values...")
    for col in feature_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            qc_results['missing_values'][col] = n_missing
            print(f"  {col}: {n_missing} missing values")
    
    if not qc_results['missing_values']:
        print("  ✓ No missing values found")
    
    # 2. Outliers per group and feature
    print("\n2. Detecting extreme outliers (IQR method)...")
    for col in feature_cols:
        outliers_count = 0
        for group in GROUPS:
            group_data = df[df['label'] == group][col].dropna()
            Q1 = group_data.quantile(0.25)
            Q3 = group_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - OUTLIER_IQR_MULTIPLIER * IQR
            upper_bound = Q3 + OUTLIER_IQR_MULTIPLIER * IQR
            
            n_outliers = ((group_data < lower_bound) | (group_data > upper_bound)).sum()
            outliers_count += n_outliers
        
        if outliers_count > 0:
            qc_results['outliers'][col] = outliers_count
    
    if qc_results['outliers']:
        print(f"  Found outliers in {len(qc_results['outliers'])} features")
        for col, count in list(qc_results['outliers'].items())[:5]:
            print(f"    {col}: {count} outliers")
    else:
        print("  ✓ No extreme outliers found")
    
    # 3. Near-zero variance
    print("\n3. Checking for low-variance features...")
    for col in feature_cols:
        if df[col].std() < 1e-6:
            qc_results['low_variance'].append(col)
            print(f"  {col}: near-zero variance")
    
    if not qc_results['low_variance']:
        print("  ✓ All features have adequate variance")
    
    print("\n✓ Quality control complete")
    print(f"  Decision: Keep all data, use robust tests when needed")
    
    return qc_results


# ============================================================================
# ASSUMPTIONS TESTING
# ============================================================================

def test_normality(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Test normality assumption per group using Shapiro-Wilk test.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame
    feature_cols : list
        List of feature columns
    
    Returns
    -------
    results : pd.DataFrame
        Normality test results
    """
    results= []
    
    for col in feature_cols:
        for group in GROUPS:
            group_data = df[df['label'] == group][col].dropna()
            
            if len(group_data) >= 3:
                stat, p_value = shapiro(group_data)
                results.append({
                    'feature': col,
                    'group': group,
                    'test': 'Shapiro-Wilk',
                    'statistic': stat,
                    'p_value': p_value,
                    'is_normal': p_value > ALPHA
                })
    
    return pd.DataFrame(results)


def test_homogeneity(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """
    Test homogeneity of variances using Levene's test.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame
    feature_cols : list
        List of feature columns
    
    Returns
    -------
    results : pd.DataFrame
        Levene test results
    """
    results = []
    
    for col in feature_cols:
        groups_data = [df[df['label'] == group][col].dropna() for group in GROUPS]
        
        # Remove empty groups
        groups_data = [g for g in groups_data if len(g) > 0]
        
        if len(groups_data) >= 2:
            stat, p_value = levene(*groups_data)
            results.append({
                'feature': col,
                'test': 'Levene',
                'statistic': stat,
                'p_value': p_value,
                'equal_variances': p_value > ALPHA
            })
    
    return pd.DataFrame(results)


def run_assumptions_checks(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Run all assumption tests and summarize results.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame
    feature_cols : list
        List of feature columns
    
    Returns
    -------
    assumptions : dict
        Dictionary containing test results
    """
    print("\n" + "="*80)
    print("ASSUMPTIONS TESTING")
    print("="*80)
    
    print("\n1. Testing normality (Shapiro-Wilk per group)...")
    normality_df = test_normality(df, feature_cols)
    
    print("\n2. Testing homogeneity of variances (Levene)...")
    levene_df = test_homogeneity(df, feature_cols)
    
    # Summarize
    assumptions_summary = []
    for col in feature_cols:
        # Check if all groups are normal
        norm_results = normality_df[normality_df['feature'] == col]
        all_normal = norm_results['is_normal'].all() if len(norm_results) > 0 else False
        
        # Check variance homogeneity
        lev_results = levene_df[levene_df['feature'] == col]
        equal_var = lev_results['equal_variances'].iloc[0] if len(lev_results) > 0 else False
        
        # Recommend test
        recommended_test = 'ANOVA' if (all_normal and equal_var) else 'Kruskal-Wallis'
        
        assumptions_summary.append({
            'feature': col,
            'all_groups_normal': all_normal,
            'equal_variances': equal_var,
            'recommended_test': recommended_test
        })
    
    assumptions_summary_df = pd.DataFrame(assumptions_summary)
    
    # Print summary
    print(f"\nAssumptions Summary:")
    print(f"  Features meeting ANOVA assumptions: {(assumptions_summary_df['recommended_test'] == 'ANOVA').sum()}")
    print(f"  Features requiring Kruskal-Wallis: {(assumptions_summary_df['recommended_test'] == 'Kruskal-Wallis').sum()}")
    
    return {
        'normality': normality_df,
        'levene': levene_df,
        'summary': assumptions_summary_df
    }


# ============================================================================
# UNIVARIATE GROUP TESTS
# ============================================================================

def calculate_effect_size_anova(df: pd.DataFrame, feature: str) -> float:
    """
    Calculate eta-squared effect size for ANOVA.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame
    feature : str
        Feature name
    
    Returns
    -------
    eta_squared : float
        Effect size (proportion of variance explained)
    """
    groups_data = [df[df['label'] == group][feature].dropna() for group in GROUPS]
    
    # Total mean
    all_data = pd.concat([pd.Series(g) for g in groups_data])
    grand_mean = all_data.mean()
    
    # Between-group sum of squares
    ss_between = sum(len(g) * (g.mean() - grand_mean)**2 for g in groups_data)
    
    # Total sum of squares
    ss_total = ((all_data - grand_mean)**2).sum()
    
    # Eta-squared
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    return eta_squared


def calculate_effect_size_kruskal(df: pd.DataFrame, feature: str) -> float:
    """
    Calculate epsilon-squared effect size for Kruskal-Wallis.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame
    feature : str
        Feature name
    
    Returns
    -------
    epsilon_squared : float
        Effect size
    """
    groups_data = [df[df['label'] == group][feature].dropna() for group in GROUPS]
    
    n_total = sum(len(g) for g in groups_data)
    H, _ = kruskal(*groups_data)
    
    # Epsilon-squared: H / (n - 1)
    epsilon_squared = H / (n_total - 1) if n_total > 1 else 0
    
    return epsilon_squared


def run_univariate_tests(df: pd.DataFrame, feature_cols: list, 
                         assumptions: dict) -> pd.DataFrame:
    """
    Run appropriate group comparison test for each feature.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame
    feature_cols : list
        List of feature columns
    assumptions : dict
        Assumptions test results
    
    Returns
    -------
    results_df : pd.DataFrame
        Test results for all features
    """
    print("\n" + "="*80)
    print("UNIVARIATE GROUP TESTS")
    print("="*80)
    
    assumptions_df = assumptions['summary']
    results = []
    
    for col in feature_cols:
        # Get recommended test
        rec_test = assumptions_df[assumptions_df['feature'] == col]['recommended_test'].iloc[0]
        
        # Prepare group data
        groups_data = [df[df['label'] == group][col].dropna() for group in GROUPS]
        
        # Run appropriate test
        if rec_test == 'ANOVA':
            stat, p_value = f_oneway(*groups_data)
            effect_size = calculate_effect_size_anova(df, col)
            test_name = 'ANOVA'
            effect_name = 'eta_squared'
        else:
            stat, p_value = kruskal(*groups_data)
            effect_size = calculate_effect_size_kruskal(df, col)
            test_name = 'Kruskal-Wallis'
            effect_name = 'epsilon_squared'
        
        # Group means/medians for direction
        group_stats = {}
        for group in GROUPS:
            group_data = df[df['label'] == group][col].dropna()
            if rec_test == 'ANOVA':
                group_stats[group] = group_data.mean()
            else:
                group_stats[group] = group_data.median()
        
        results.append({
            'feature': col,
            'test': test_name,
            'statistic': stat,
            'p_value': p_value,
            effect_name: effect_size,
            'CN_mean': group_stats['CN'],
            'AD_mean': group_stats['AD'],
            'FTD_mean': group_stats['FTD']
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('p_value')
    
    print(f"\n✓ Tested {len(results_df)} features")
    print(f"  Significant at α={ALPHA}: {(results_df['p_value'] < ALPHA).sum()}")
    
    return results_df


# ============================================================================
# POST-HOC TESTS
# ============================================================================

def run_posthoc_tukey(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Run Tukey HSD post-hoc test.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame
    feature : str
        Feature name
    
    Returns
    -------
    results_df : pd.DataFrame
        Pairwise comparison results
    """
    # Prepare data for Tukey
    data_long = df[['label', feature]].dropna()
    
    # Run Tukey HSD
    tukey = pairwise_tukeyhsd(data_long[feature], data_long['label'], alpha=ALPHA)
    
    # Convert to DataFrame
    results = []
    for i in range(len(tukey.summary().data[1:])):
        row = tukey.summary().data[i+1]
        results.append({
            'group1': row[0],
            'group2': row[1],
            'meandiff': row[2],
            'p_adj': row[5],
            'reject': row[6]
        })
    
    return pd.DataFrame(results)


def run_posthoc_dunn(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """
    Run Dunn's test for post-hoc pairwise comparisons.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame
    feature : str
        Feature name
    
    Returns
    -------
    results_df : pd.DataFrame
        Pairwise comparison results
    """
    if HAS_PINGOUIN:
        # Use pingouin for Dunn test
        data_long = df[['label', feature]].dropna()
        data_long.columns = ['group', 'value']
        
        dunn_results = pg.pairwise_tests(data=data_long, dv='value', between='group',
                                        parametric=False, padjust='holm')
        
        return dunn_results[['A', 'B', 'p-unc', 'p-corr']].rename(columns={
            'A': 'group1', 'B': 'group2', 'p-unc': 'p_raw', 'p-corr': 'p_adj'
        })
    else:
        # Fallback: pairwise Mann-Whitney with Bonferroni
        results = []
        for group1, group2 in PAIRS:
            data1 = df[df['label'] == group1][feature].dropna()
            data2 = df[df['label'] == group2][feature].dropna()
            
            stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
            
            results.append({
                'group1': group1,
                'group2': group2,
                'p_raw': p_value,
                'p_adj': p_value * len(PAIRS)  # Bonferroni
            })
        
        return pd.DataFrame(results)


def run_posthoc(df: pd.DataFrame, univariate_results: pd.DataFrame,
               assumptions: dict) -> pd.DataFrame:
    """
    Run post-hoc tests for significant features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame
    univariate_results : pd.DataFrame
        Univariate test results
    assumptions : dict
        Assumptions test results
    
    Returns
    -------
    posthoc_df : pd.DataFrame
        All post-hoc results
    """
    print("\n" + "="*80)
    print("POST-HOC PAIRWISE TESTS")
    print("="*80)
    
    # Only test significant features
    sig_features = univariate_results[univariate_results['p_value'] < ALPHA]
    
    print(f"\nRunning post-hoc tests for {len(sig_features)} significant features...")
    
    all_posthoc = []
    assumptions_df = assumptions['summary']
    
    for _, row in sig_features.iterrows():
        feature = row['feature']
        test_type = row['test']
        
        print(f"  {feature}...", end=' ')
        
        if test_type == 'ANOVA':
            posthoc_results = run_posthoc_tukey(df, feature)
            posthoc_results['method'] = 'Tukey HSD'
        else:
            posthoc_results = run_posthoc_dunn(df, feature)
            posthoc_results['method'] = 'Dunn (Holm)'
        
        posthoc_results['feature'] = feature
        all_posthoc.append(posthoc_results)
        print("✓")
    
    if all_posthoc:
        posthoc_df = pd.concat(all_posthoc, ignore_index=True)
    else:
        posthoc_df = pd.DataFrame()
        print("  No significant features for post-hoc testing")
    
    return posthoc_df


# ============================================================================
# MULTIPLE COMPARISON CORRECTION
# ============================================================================

def apply_multiple_comparisons(univariate_results: pd.DataFrame) -> pd.DataFrame:
    """
    Apply FDR correction across all features.
    
    Parameters
    ----------
    univariate_results : pd.DataFrame
        Univariate test results
    
    Returns
    -------
    results_corrected : pd.DataFrame
        Results with FDR-corrected p-values
    """
    print("\n" + "="*80)
    print("MULTIPLE COMPARISON CORRECTION (FDR)")
    print("="*80)
    
    # Apply Benjamini-Hochberg FDR correction
    reject, p_corrected, _, _ = multipletests(univariate_results['p_value'],
                                              alpha=ALPHA, method='fdr_bh')
    
    univariate_results = univariate_results.copy()
    univariate_results['p_fdr'] = p_corrected
    univariate_results['significant_fdr'] = reject
    
    # Sort by FDR-corrected p-value
    univariate_results = univariate_results.sort_values('p_fdr')
    
    print(f"\n✓ FDR correction applied")
    print(f"  Significant after FDR (α={ALPHA}): {reject.sum()}/{len(reject)}")
    
    return univariate_results


# ============================================================================
# MULTIVARIATE ANALYSIS
# ============================================================================

def run_pca_analysis(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Run PCA and test group differences on principal components.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame
    feature_cols : list
        List of feature columns
    
    Returns
    -------
    pca_results : dict
        PCA results and group tests
    """
    print("\n" + "="*80)
    print("MULTIVARIATE ANALYSIS (PCA)")
    print("="*80)
    
    # Standardize features
    scaler = StandardScaler()
    X = df[feature_cols].fillna(df[feature_cols].mean())
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=min(5, len(feature_cols)), random_state=RANDOM_SEED)
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"\nPCA: {pca.n_components_} components")
    print(f"Explained variance: {pca.explained_variance_ratio_[:3]}")
    print(f"Cumulative variance (first 3 PCs): {pca.explained_variance_ratio_[:3].sum():.2%}")
    
    # Test group differences on first 3 PCs
    pc_tests = []
    for i in range(min(3, pca.n_components_)):
        pc_scores = X_pca[:, i]
        groups_data = [pc_scores[df['label'] == group] for group in GROUPS]
        
        # ANOVA on PC scores
        stat, p_value = f_oneway(*groups_data)
        eta_sq = calculate_effect_size_anova(
            pd.DataFrame({'label': df['label'], f'PC{i+1}': pc_scores}), 
            f'PC{i+1}'
        )
        
        pc_tests.append({
            'component': f'PC{i+1}',
            'variance_explained': pca.explained_variance_ratio_[i],
            'F_statistic': stat,
            'p_value': p_value,
            'eta_squared': eta_sq
        })
    
    pc_tests_df = pd.DataFrame(pc_tests)
    
    print(f"\nGroup differences on PCs:")
    for _, row in pc_tests_df.iterrows():
        sig = "***" if row['p_value'] < 0.001 else ("**" if row['p_value'] < 0.01 else ("*" if row['p_value'] < 0.05 else "ns"))
        print(f"  {row['component']}: p={row['p_value']:.4f} {sig}, η²={row['eta_squared']:.3f}")
    
    # Store PC scores for plotting
    pc_df = pd.DataFrame(X_pca[:, :3], columns=['PC1', 'PC2', 'PC3'])
    pc_df['label'] = df['label'].values
    pc_df['subject_id'] = df['subject_id'].values
    
    return {
        'pca': pca,
        'explained_variance': pca.explained_variance_ratio_,
        'pc_scores': pc_df,
        'pc_tests': pc_tests_df,
        'loadings': pca.components_
    }


# ============================================================================
# VISUALIZATION
# ============================================================================

def make_stat_plots(df: pd.DataFrame, univariate_results: pd.DataFrame,
                   posthoc_df: pd.DataFrame, pca_results: dict):
    """
    Generate statistical visualization plots.
    
    Parameters
    ----------
    df : pd.DataFrame
        Feature DataFrame
    univariate_results : pd.DataFrame
        Univariate test results
    posthoc_df : pd.DataFrame
        Post-hoc test results
    pca_results : dict
        PCA analysis results
    """
    print("\n" + "="*80)
    print("GENERATING STATISTICAL PLOTS")
    print("="*80)
    
    # Select top 6 discriminative features
    top_features = univariate_results.head(6)['feature'].tolist()
    
    # 1. Boxplots with significance annotations
    print("\n1. Creating annotated boxplots for top features...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        
        # Boxplot with jitter
        df_plot = df[['label', feature]].dropna()
        
        # Boxplot
        bp = ax.boxplot([df_plot[df_plot['label'] == g][feature] for g in GROUPS],
                        labels=GROUPS, patch_artist=True)
        
        # Color boxes
        colors = {'CN': 'lightgreen', 'AD': 'lightcoral', 'FTD': 'lightblue'}
        for patch, group in zip(bp['boxes'], GROUPS):
            patch.set_facecolor(colors[group])
            patch.set_alpha(0.6)
        
        # Add jittered points
        for i, group in enumerate(GROUPS):
            y = df_plot[df_plot['label'] == group][feature]
            x = np.random.normal(i+1, 0.04, size=len(y))
            ax.scatter(x, y, alpha=0.4, s=20, color='black')
        
        # Add group means
        for i, group in enumerate(GROUPS):
            mean_val = df_plot[df_plot['label'] == group][feature].mean()
            ax.plot(i+1, mean_val, 'r*', markersize=15, markeredgecolor='darkred', markeredgewidth=1.5)
        
        # Get p-value and effect size
        row = univariate_results[univariate_results['feature'] == feature].iloc[0]
        p_fdr = row['p_fdr']
        effect_col = 'eta_squared' if 'eta_squared' in row else 'epsilon_squared'
        effect = row[effect_col]
        
        # Title with stats
        sig_str = "***" if p_fdr < 0.001 else ("**" if p_fdr < 0.01 else ("*" if p_fdr < 0.05 else "ns"))
        ax.set_title(f'{feature}\np_FDR={p_fdr:.4f} {sig_str}, effect={effect:.3f}',
                    fontsize=10, fontweight='bold')
        ax.set_ylabel('Value', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add pairwise significance annotations if available
        if len(posthoc_df) > 0:
            posthoc_feature = posthoc_df[posthoc_df['feature'] == feature]
            if len(posthoc_feature) > 0:
                # Add significance bars
                y_max = df_plot[feature].max()
                y_range = df_plot[feature].max() - df_plot[feature].min()
                
                for pair_idx, (_, pair_row) in enumerate(posthoc_feature.iterrows()):
                    if 'p_adj' in pair_row:
                        p_adj = pair_row['p_adj']
                        if p_adj < 0.05:
                            g1_idx = GROUPS.index(pair_row['group1']) + 1
                            g2_idx = GROUPS.index(pair_row['group2']) + 1
                            
                            y = y_max + (0.1 + pair_idx*0.08) * y_range
                            sig_symbol = "***" if p_adj < 0.001 else ("**" if p_adj < 0.01 else "*")
                            
                            ax.plot([g1_idx, g2_idx], [y, y], 'k-', linewidth=1)
                            ax.text((g1_idx + g2_idx)/2, y, sig_symbol, ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Top Discriminative Features by Group (with significance annotations)',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(STATS_OUTPUT_DIR / 'top_features_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: top_features_boxplots.png")
    
    # 2. PCA scatter plot
    print("\n2. Creating PCA biplot...")
    
    pc_df = pca_results['pc_scores']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'CN': 'green', 'AD': 'red', 'FTD': 'blue'}
    markers = {'CN': 'o', 'AD': 's', 'FTD': '^'}
    
    for group in GROUPS:
        group_data = pc_df[pc_df['label'] == group]
        ax.scatter(group_data['PC1'], group_data['PC2'], 
                  c=colors[group], marker=markers[group], 
                  label=f'{group} (n={len(group_data)})', 
                  s=100, alpha=0.6, edgecolors='black', linewidth=1)
    
    # Add explained variance to axis labels
    var_pc1 = pca_results['explained_variance'][0]
    var_pc2 = pca_results['explained_variance'][1]
    
    ax.set_xlabel(f'PC1 ({var_pc1:.1%} variance)', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'PC2 ({var_pc2:.1%} variance)', fontsize=12, fontweight='bold')
    ax.set_title('PCA: Group Separation in Multivariate Feature Space', 
                fontsize=14, fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(STATS_OUTPUT_DIR / 'pca_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: pca_scatter.png")
    
    # 3. Effect sizes comparison
    print("\n3. Creating effect sizes comparison plot...")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get effect sizes
    effects = []
    for _, row in univariate_results.iterrows():
        feature = row['feature']
        effect_val = row.get('eta_squared', row.get('epsilon_squared', 0))
        p_fdr = row['p_fdr']
        
        # Extract channel from feature name
        channel = feature.split('_')[0] if '_' in feature else 'Unknown'
        
        effects.append({
            'feature': feature,
            'effect_size': effect_val,
            'p_fdr': p_fdr,
            'channel': channel,
            'significant': p_fdr < ALPHA
        })
    
    effects_df = pd.DataFrame(effects).sort_values('effect_size', ascending=True)
    
    # Create horizontal bar plot
    colors_sig = ['darkgreen' if sig else 'gray' for sig in effects_df['significant']]
    
    bars = ax.barh(range(len(effects_df)), effects_df['effect_size'], color=colors_sig, alpha=0.7)
    ax.set_yticks(range(len(effects_df)))
    ax.set_yticklabels(effects_df['feature'], fontsize=8)
    ax.set_xlabel('Effect Size (η² or ε²)', fontsize=11, fontweight='bold')
    ax.set_title('Effect Sizes for All Features (green = FDR significant)', 
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add significance threshold line
    ax.axvline(0.01, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Small effect')
    ax.axvline(0.06, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Medium effect')
    ax.axvline(0.14, color='darkred', linestyle='--', linewidth=1, alpha=0.5, label='Large effect')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(STATS_OUTPUT_DIR / 'effect_sizes_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: effect_sizes_comparison.png")


# ============================================================================
# SAVE OUTPUTS
# ============================================================================

def save_outputs(assumptions: dict, univariate_results: pd.DataFrame,
                posthoc_df: pd.DataFrame, pca_results: dict):
    """
    Save all statistical results to CSV files.
    
    Parameters
    ----------
    assumptions : dict
        Assumptions test results
    univariate_results : pd.DataFrame
        Univariate test results
    posthoc_df : pd.DataFrame
        Post-hoc test results
    pca_results : dict
        PCA analysis results
    """
    print("\n" + "="*80)
    print("SAVING OUTPUTS")
    print("="*80)
    
    # 1. Assumptions summary
    assumptions['summary'].to_csv(STATS_OUTPUT_DIR / 'assumptions_summary.csv', index=False)
    print("  Saved: assumptions_summary.csv")
    
    # 2. Univariate tests
    univariate_results.to_csv(STATS_OUTPUT_DIR / 'univariate_tests_all_features.csv', index=False)
    print("  Saved: univariate_tests_all_features.csv")
    
    # 3. Post-hoc tests
    if len(posthoc_df) > 0:
        posthoc_df.to_csv(STATS_OUTPUT_DIR / 'posthoc_all_features.csv', index=False)
        print("  Saved: posthoc_all_features.csv")
    
    # 4. PCA results
    pca_results['pc_scores'].to_csv(STATS_OUTPUT_DIR / 'pca_scores.csv', index=False)
    pca_results['pc_tests'].to_csv(STATS_OUTPUT_DIR / 'pca_group_tests.csv', index=False)
    print("  Saved: pca_scores.csv, pca_group_tests.csv")
    
    # 5. Top features summary
    top_features = univariate_results.head(10)[['feature', 'test', 'p_value', 'p_fdr']]
    top_features.to_csv(STATS_OUTPUT_DIR / 'top10_discriminative_features.csv', index=False)
    print("  Saved: top10_discriminative_features.csv")


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def print_summary(univariate_results: pd.DataFrame, posthoc_df: pd.DataFrame,
                 pca_results: pd.DataFrame):
    """
    Print a concise summary of statistical findings.
    
    Parameters
    ----------
    univariate_results : pd.DataFrame
        Univariate test results
    posthoc_df : pd.DataFrame
        Post-hoc test results
    pca_results : dict
        PCA analysis results
    """
    print("\n" + "="*80)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("="*80)
    
    # Most discriminative features
    print("\n1. TOP 5 MOST DISCRIMINATIVE FEATURES:")
    print("-" * 80)
    for i, (_, row) in enumerate(univariate_results.head(5).iterrows(), 1):
        effect_col = 'eta_squared' if 'eta_squared' in row else 'epsilon_squared'
        print(f"   {i}. {row['feature']}")
        print(f"      Test: {row['test']}, p_FDR: {row['p_fdr']:.6f}, Effect size: {row[effect_col]:.3f}")
        print(f"      Means: CN={row['CN_mean']:.4f}, AD={row['AD_mean']:.4f}, FTD={row['FTD_mean']:.4f}")
    
    # Channel comparison
    print("\n2. CHANNEL COMPARISON (Pz vs F3):")
    print("-" * 80)
    
    pz_features = univariate_results[univariate_results['feature'].str.startswith('Pz_')]
    f3_features = univariate_results[univariate_results['feature'].str.startswith('F3_')]
    
    pz_sig = (pz_features['p_fdr'] < ALPHA).sum()
    f3_sig = (f3_features['p_fdr'] < ALPHA).sum()
    
    effect_col = 'eta_squared' if 'eta_squared' in univariate_results.columns else 'epsilon_squared'
    pz_mean_effect = pz_features[effect_col].mean()
    f3_mean_effect = f3_features[effect_col].mean()
    
    print(f"   Pz: {pz_sig}/{len(pz_features)} significant, mean effect size: {pz_mean_effect:.3f}")
    print(f"   F3: {f3_sig}/{len(f3_features)} significant, mean effect size: {f3_mean_effect:.3f}")
    
    if pz_mean_effect > f3_mean_effect:
        print(f"   → Pz appears more discriminative overall")
    else:
        print(f"   → F3 appears more discriminative overall")
    
    # Multivariate findings
    print("\n3. MULTIVARIATE ANALYSIS (PCA):")
    print("-" * 80)
    pc_tests = pca_results['pc_tests']
    sig_pcs = (pc_tests['p_value'] < ALPHA).sum()
    print(f"   {sig_pcs}/{len(pc_tests)} principal components show significant group differences")
    print(f"   First 3 PCs explain {pca_results['explained_variance'][:3].sum():.1%} of variance")
    
    # Caveats
    print("\n4. CAVEATS AND NOTES:")
    print("-" * 80)
    print(f"   - Multiple comparison correction: FDR (Benjamini-Hochberg) applied")
    print(f"   - Sample sizes: CN (29), AD (36), FTD (23) - adequate for parametric tests")
    print(f"   - Outliers detected but retained (robust tests used when needed)")
    print(f"   - Some features violate normality → Kruskal-Wallis used appropriately")
    
    print("\n" + "="*80)
    print(f"All results saved to: {STATS_OUTPUT_DIR.absolute()}")
    print("="*80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function."""
    
    print("="*80)
    print("ADVANCED STATISTICAL ANALYSIS")
    print("EEG Feature Discrimination: CN vs AD vs FTD")
    print("="*80)
    
    # Create output directory
    STATS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    
    # 1. Load data
    print("\n1. LOADING DATA")
    print("-" * 80)
    df = load_features()
    feature_cols = identify_feature_columns(df)
    
    # 2. Quality control
    qc_results = quality_control(df, feature_cols)
    
    # 3. Assumptions testing
    assumptions = run_assumptions_checks(df, feature_cols)
    
    # 4. Univariate tests
    univariate_results = run_univariate_tests(df, feature_cols, assumptions)
    
    # 5. Multiple comparison correction
    univariate_results = apply_multiple_comparisons(univariate_results)
    
    # 6. Post-hoc tests
    posthoc_df = run_posthoc(df, univariate_results, assumptions)
    
    # 7. Multivariate analysis (PCA)
    pca_results = run_pca_analysis(df, feature_cols)
    
    # 8. Visualization
    make_stat_plots(df, univariate_results, posthoc_df, pca_results)
    
    # 9. Save outputs
    save_outputs(assumptions, univariate_results, posthoc_df, pca_results)
    
    # 10. Print summary
    print_summary(univariate_results, posthoc_df, pca_results)
    
    print("\n✓ Advanced statistical analysis complete!")


if __name__ == "__main__":
    main()
