"""
3-Class Classification: 2-Channel vs 4-Channel Comparison
==========================================================

Testing if F4 and O1 help differentiate AD vs FTD
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, balanced_accuracy_score, 
                            classification_report, f1_score)
import warnings
warnings.filterwarnings('ignore')

# Paths
FEATURES_FILE_2CH = Path('results/part2/simple_stats/features.csv')
FEATURES_FILE_4CH = Path('results/part2/simple_stats_4ch/features.csv')
OUTPUT_DIR = Path('results/part2/4channel_comparison')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
CV_FOLDS = 5

print("="*80)
print("3-CLASS CLASSIFICATION: 2-CHANNEL vs 4-CHANNEL")
print("="*80)

# Load both datasets
df_2ch = pd.read_csv(FEATURES_FILE_2CH)
df_4ch = pd.read_csv(FEATURES_FILE_4CH)

# Map labels to integers
label_map = {'CN': 0, 'AD': 1, 'FTD': 2}
df_2ch['label_int'] = df_2ch['label'].map(label_map)
df_4ch['label_int'] = df_4ch['label'].map(label_map)

# Extract features
metadata_cols = ['subject_id', 'label', 'label_int']
features_2ch = [col for col in df_2ch.columns if col not in metadata_cols]
features_4ch = [col for col in df_4ch.columns if col not in metadata_cols]

print(f"\nDataset: {len(df_2ch)} subjects")
print(f"  CN:  {(df_2ch['label'] == 'CN').sum()}")
print(f"  AD:  {(df_2ch['label'] == 'AD').sum()}")
print(f"  FTD: {(df_2ch['label'] == 'FTD').sum()}")

print(f"\n2-Channel: {len(features_2ch)} features (Pz, F3)")
print(f"4-Channel: {len(features_4ch)} features (Pz, F3, F4, O1)")

# Prepare data
X_2ch = df_2ch[features_2ch].values
X_4ch = df_4ch[features_4ch].values
y = df_4ch['label_int'].values

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

print("\n" + "="*80)
print("TESTING MODELS ON 3-CLASS PROBLEM")
print("="*80)

# Models
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            C=1, 
            class_weight='balanced',
            max_iter=1000,
            random_state=RANDOM_SEED
        ))
    ]),
    'SVM (RBF)': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(
            kernel='rbf',
            C=1,
            gamma='scale',
            class_weight='balanced',
            random_state=RANDOM_SEED
        ))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            class_weight='balanced_subsample',
            random_state=RANDOM_SEED
        ))
    ])
}

results = []

for model_name, model in models.items():
    print(f"\n{model_name}:")
    print("-" * 60)
    
    # 2-Channel
    print("\n  2-Channel (Pz, F3):")
    y_pred_2ch = cross_val_predict(model, X_2ch, y, cv=cv)
    bal_acc_2ch = balanced_accuracy_score(y, y_pred_2ch)
    f1_2ch = f1_score(y, y_pred_2ch, average='macro')
    
    cm_2ch = confusion_matrix(y, y_pred_2ch)
    
    # Per-class recall
    recall_CN_2ch = cm_2ch[0,0] / cm_2ch[0,:].sum()
    recall_AD_2ch = cm_2ch[1,1] / cm_2ch[1,:].sum()
    recall_FTD_2ch = cm_2ch[2,2] / cm_2ch[2,:].sum()
    
    print(f"    Balanced Accuracy: {bal_acc_2ch:.4f}")
    print(f"    F1-Score (macro):  {f1_2ch:.4f}")
    print(f"    Recall CN:  {recall_CN_2ch:.4f}")
    print(f"    Recall AD:  {recall_AD_2ch:.4f}")
    print(f"    Recall FTD: {recall_FTD_2ch:.4f}")
    
    # 4-Channel
    print("\n  4-Channel (Pz, F3, F4, O1):")
    y_pred_4ch = cross_val_predict(model, X_4ch, y, cv=cv)
    bal_acc_4ch = balanced_accuracy_score(y, y_pred_4ch)
    f1_4ch = f1_score(y, y_pred_4ch, average='macro')
    
    cm_4ch = confusion_matrix(y, y_pred_4ch)
    
    # Per-class recall
    recall_CN_4ch = cm_4ch[0,0] / cm_4ch[0,:].sum()
    recall_AD_4ch = cm_4ch[1,1] / cm_4ch[1,:].sum()
    recall_FTD_4ch = cm_4ch[2,2] / cm_4ch[2,:].sum()
    
    print(f"    Balanced Accuracy: {bal_acc_4ch:.4f}")
    print(f"    F1-Score (macro):  {f1_4ch:.4f}")
    print(f"    Recall CN:  {recall_CN_4ch:.4f}")
    print(f"    Recall AD:  {recall_AD_4ch:.4f}")
    print(f"    Recall FTD: {recall_FTD_4ch:.4f}")
    
    # Improvement
    improvement = ((bal_acc_4ch - bal_acc_2ch) / bal_acc_2ch) * 100
    ftd_improvement = ((recall_FTD_4ch - recall_FTD_2ch) / recall_FTD_2ch) * 100 if recall_FTD_2ch > 0 else float('inf')
    
    print(f"\n  📊 Overall Improvement: {improvement:+.1f}%")
    print(f"  🎯 FTD Recall Improvement: {ftd_improvement:+.1f}%")
    
    results.append({
        'model': model_name,
        'channels': '2-ch',
        'balanced_accuracy': bal_acc_2ch,
        'f1_macro': f1_2ch,
        'recall_CN': recall_CN_2ch,
        'recall_AD': recall_AD_2ch,
        'recall_FTD': recall_FTD_2ch
    })
    
    results.append({
        'model': model_name,
        'channels': '4-ch',
        'balanced_accuracy': bal_acc_4ch,
        'f1_macro': f1_4ch,
        'recall_CN': recall_CN_4ch,
        'recall_AD': recall_AD_4ch,
        'recall_FTD': recall_FTD_4ch
    })

# Results dataframe
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / '3class_2ch_vs_4ch.csv', index=False)

print("\n" + "="*80)
print("SUMMARY: 3-CLASS PERFORMANCE")
print("="*80)

print("\nBalanced Accuracy:")
for model_name in models.keys():
    bal_2ch = results_df[(results_df['model'] == model_name) & (results_df['channels'] == '2-ch')]['balanced_accuracy'].values[0]
    bal_4ch = results_df[(results_df['model'] == model_name) & (results_df['channels'] == '4-ch')]['balanced_accuracy'].values[0]
    improvement = ((bal_4ch - bal_2ch) / bal_2ch) * 100
    print(f"  {model_name:25s}: {bal_2ch:.4f} → {bal_4ch:.4f} ({improvement:+.1f}%)")

print("\nFTD Recall (Critical Metric):")
for model_name in models.keys():
    ftd_2ch = results_df[(results_df['model'] == model_name) & (results_df['channels'] == '2-ch')]['recall_FTD'].values[0]
    ftd_4ch = results_df[(results_df['model'] == model_name) & (results_df['channels'] == '4-ch')]['recall_FTD'].values[0]
    ftd_improvement = ((ftd_4ch - ftd_2ch) / ftd_2ch) * 100 if ftd_2ch > 0 else float('inf')
    print(f"  {model_name:25s}: {ftd_2ch:.4f} → {ftd_4ch:.4f} ({ftd_improvement:+.1f}%)")

# Best overall
best_row = results_df.loc[results_df['balanced_accuracy'].idxmax()]
print(f"\n🏆 Best Overall:")
print(f"   {best_row['model']} with {best_row['channels']}")
print(f"   Balanced Accuracy: {best_row['balanced_accuracy']:.4f}")
print(f"   F1-Score: {best_row['f1_macro']:.4f}")
print(f"   FTD Recall: {best_row['recall_FTD']:.4f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Balanced Accuracy
ax = axes[0]
models_list = list(models.keys())
x = np.arange(len(models_list))
width = 0.35

bal_2ch_list = [results_df[(results_df['model'] == m) & (results_df['channels'] == '2-ch')]['balanced_accuracy'].values[0] for m in models_list]
bal_4ch_list = [results_df[(results_df['model'] == m) & (results_df['channels'] == '4-ch')]['balanced_accuracy'].values[0] for m in models_list]

bars1 = ax.bar(x - width/2, bal_2ch_list, width, label='2-ch', alpha=0.8, color='steelblue')
bars2 = ax.bar(x + width/2, bal_4ch_list, width, label='4-ch', alpha=0.8, color='coral')

ax.set_ylabel('Balanced Accuracy', fontweight='bold')
ax.set_title('3-Class: 2-ch vs 4-ch Comparison', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_list, rotation=15, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0.3, 0.6])

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

# Plot 2: FTD Recall (Critical!)
ax = axes[1]
ftd_2ch_list = [results_df[(results_df['model'] == m) & (results_df['channels'] == '2-ch')]['recall_FTD'].values[0] for m in models_list]
ftd_4ch_list = [results_df[(results_df['model'] == m) & (results_df['channels'] == '4-ch')]['recall_FTD'].values[0] for m in models_list]

bars1 = ax.bar(x - width/2, ftd_2ch_list, width, label='2-ch', alpha=0.8, color='steelblue')
bars2 = ax.bar(x + width/2, ftd_4ch_list, width, label='4-ch', alpha=0.8, color='coral')

ax.set_ylabel('FTD Recall', fontweight='bold')
ax.set_title('FTD Detection: 2-ch vs 4-ch', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_list, rotation=15, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(0.33, color='red', linestyle='--', label='Chance (33%)', alpha=0.5)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '3class_2ch_vs_4ch.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {OUTPUT_DIR / '3class_2ch_vs_4ch.png'}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
