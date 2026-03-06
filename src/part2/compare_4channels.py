"""
4-Channel Binary Classification: CN vs Dementia
================================================

Testing if adding more channels (Pz, F3, F4, O1) improves performance
compared to 2-channel (Pz, F3) baseline.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, balanced_accuracy_score, f1_score, roc_auc_score)
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
print("4-CHANNEL BINARY CLASSIFICATION COMPARISON")
print("="*80)

# Load both datasets
df_2ch = pd.read_csv(FEATURES_FILE_2CH)
df_4ch = pd.read_csv(FEATURES_FILE_4CH)

# Create binary labels
df_2ch['binary_label'] = df_2ch['label'].map({'CN': 0, 'AD': 1, 'FTD': 1})
df_4ch['binary_label'] = df_4ch['label'].map({'CN': 0, 'AD': 1, 'FTD': 1})

# Extract features
metadata_cols = ['subject_id', 'label', 'binary_label']
features_2ch = [col for col in df_2ch.columns if col not in metadata_cols]
features_4ch = [col for col in df_4ch.columns if col not in metadata_cols]

print(f"\n2-Channel Dataset (Pz, F3):")
print(f"  Features: {len(features_2ch)}")
print(f"  Channels: Pz, F3")

print(f"\n4-Channel Dataset (Pz, F3, F4, O1):")
print(f"  Features: {len(features_4ch)}")
print(f"  Channels: Pz, F3, F4, O1")

# Prepare data
X_2ch = df_2ch[features_2ch].values
X_4ch = df_4ch[features_4ch].values
y = df_4ch['binary_label'].values

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

print("\n" + "="*80)
print("TESTING OPTIMIZED MODELS ON BOTH DATASETS")
print("="*80)

results = []

# Define models to test (using best hyperparameters from 2-channel optimization)
models = {
    'Logistic Regression': {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                C=0.01, 
                class_weight='balanced',
                max_iter=1000, 
                random_state=RANDOM_SEED
            ))
        ]),
        'description': 'Best from 2-ch optimization (C=0.01, balanced)'
    },
    'SVM (RBF)': {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(
                kernel='rbf',
                C=1,
                gamma=0.01,
                class_weight='balanced',
                probability=True,
                random_state=RANDOM_SEED
            ))
        ]),
        'description': 'Best from 2-ch optimization (C=1, gamma=0.01, balanced)'
    },
    'Random Forest': {
        'model': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=3,
                min_samples_split=5,
                class_weight='balanced_subsample',
                random_state=RANDOM_SEED
            ))
        ]),
        'description': 'Best from 2-ch optimization (depth=3, balanced_subsample)'
    }
}

for model_name, model_info in models.items():
    print(f"\n{model_name}:")
    print("-" * 60)
    print(f"  {model_info['description']}")
    
    model = model_info['model']
    
    # Evaluate on 2-channel
    print("\n  2-Channel Results:")
    bal_acc_2ch = cross_val_score(model, X_2ch, y, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
    f1_2ch = cross_val_score(model, X_2ch, y, cv=cv, scoring='f1', n_jobs=-1)
    roc_auc_2ch = cross_val_score(model, X_2ch, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    y_pred_2ch = cross_val_predict(model, X_2ch, y, cv=cv)
    cm_2ch = confusion_matrix(y, y_pred_2ch)
    TN, FP, FN, TP = cm_2ch[0,0], cm_2ch[0,1], cm_2ch[1,0], cm_2ch[1,1]
    sens_2ch = TP / (TP + FN)
    spec_2ch = TN / (TN + FP)
    
    print(f"    Balanced Accuracy: {bal_acc_2ch.mean():.4f} ± {bal_acc_2ch.std():.4f}")
    print(f"    F1-Score:          {f1_2ch.mean():.4f} ± {f1_2ch.std():.4f}")
    print(f"    ROC AUC:           {roc_auc_2ch.mean():.4f} ± {roc_auc_2ch.std():.4f}")
    print(f"    Sensitivity:       {sens_2ch:.4f}")
    print(f"    Specificity:       {spec_2ch:.4f}")
    
    # Evaluate on 4-channel
    print("\n  4-Channel Results:")
    bal_acc_4ch = cross_val_score(model, X_4ch, y, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
    f1_4ch = cross_val_score(model, X_4ch, y, cv=cv, scoring='f1', n_jobs=-1)
    roc_auc_4ch = cross_val_score(model, X_4ch, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    y_pred_4ch = cross_val_predict(model, X_4ch, y, cv=cv)
    cm_4ch = confusion_matrix(y, y_pred_4ch)
    TN, FP, FN, TP = cm_4ch[0,0], cm_4ch[0,1], cm_4ch[1,0], cm_4ch[1,1]
    sens_4ch = TP / (TP + FN)
    spec_4ch = TN / (TN + FP)
    
    print(f"    Balanced Accuracy: {bal_acc_4ch.mean():.4f} ± {bal_acc_4ch.std():.4f}")
    print(f"    F1-Score:          {f1_4ch.mean():.4f} ± {f1_4ch.std():.4f}")
    print(f"    ROC AUC:           {roc_auc_4ch.mean():.4f} ± {roc_auc_4ch.std():.4f}")
    print(f"    Sensitivity:       {sens_4ch:.4f}")
    print(f"    Specificity:       {spec_4ch:.4f}")
    
    # Improvement
    improvement = ((bal_acc_4ch.mean() - bal_acc_2ch.mean()) / bal_acc_2ch.mean()) * 100
    print(f"\n  📊 Improvement: {improvement:+.1f}%")
    
    results.append({
        'model': model_name,
        'channels': '2-ch (Pz, F3)',
        'balanced_accuracy': bal_acc_2ch.mean(),
        'bal_acc_std': bal_acc_2ch.std(),
        'f1_score': f1_2ch.mean(),
        'roc_auc': roc_auc_2ch.mean(),
        'sensitivity': sens_2ch,
        'specificity': spec_2ch
    })
    
    results.append({
        'model': model_name,
        'channels': '4-ch (Pz, F3, F4, O1)',
        'balanced_accuracy': bal_acc_4ch.mean(),
        'bal_acc_std': bal_acc_4ch.std(),
        'f1_score': f1_4ch.mean(),
        'roc_auc': roc_auc_4ch.mean(),
        'sensitivity': sens_4ch,
        'specificity': spec_4ch
    })

# Results dataframe
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_DIR / '2ch_vs_4ch_comparison.csv', index=False)

print("\n" + "="*80)
print("SUMMARY COMPARISON")
print("="*80)

print("\nBalanced Accuracy by Model and Channels:")
pivot = results_df.pivot(index='model', columns='channels', values='balanced_accuracy')
print(pivot.to_string())

print("\nAverage improvement from 2-ch to 4-ch:")
for model_name in models.keys():
    bal_2ch = results_df[(results_df['model'] == model_name) & (results_df['channels'] == '2-ch (Pz, F3)')]['balanced_accuracy'].values[0]
    bal_4ch = results_df[(results_df['model'] == model_name) & (results_df['channels'] == '4-ch (Pz, F3, F4, O1)')]['balanced_accuracy'].values[0]
    improvement = ((bal_4ch - bal_2ch) / bal_2ch) * 100
    print(f"  {model_name}: {improvement:+.1f}% ({bal_2ch:.4f} → {bal_4ch:.4f})")

# Best overall
best_row = results_df.loc[results_df['balanced_accuracy'].idxmax()]
print(f"\n🏆 Best Overall:")
print(f"   {best_row['model']} with {best_row['channels']}")
print(f"   Balanced Accuracy: {best_row['balanced_accuracy']:.4f}")
print(f"   F1-Score: {best_row['f1_score']:.4f}")
print(f"   ROC AUC: {best_row['roc_auc']:.4f}")
print(f"   Sensitivity: {best_row['sensitivity']:.4f}")
print(f"   Specificity: {best_row['specificity']:.4f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Balanced Accuracy comparison
ax = axes[0]
models_list = list(models.keys())
x = np.arange(len(models_list))
width = 0.35

bal_acc_2ch_list = [results_df[(results_df['model'] == m) & (results_df['channels'] == '2-ch (Pz, F3)')]['balanced_accuracy'].values[0] for m in models_list]
bal_acc_4ch_list = [results_df[(results_df['model'] == m) & (results_df['channels'] == '4-ch (Pz, F3, F4, O1)')]['balanced_accuracy'].values[0] for m in models_list]

bars1 = ax.bar(x - width/2, bal_acc_2ch_list, width, label='2-ch (Pz, F3)', alpha=0.8, color='steelblue')
bars2 = ax.bar(x + width/2, bal_acc_4ch_list, width, label='4-ch (Pz, F3, F4, O1)', alpha=0.8, color='coral')

ax.set_ylabel('Balanced Accuracy', fontweight='bold')
ax.set_title('2-Channel vs 4-Channel Comparison', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models_list, rotation=15, ha='right')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0.5, 0.85])

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

# Plot 2: Improvement percentages
ax = axes[1]
improvements = []
for model_name in models_list:
    bal_2ch = results_df[(results_df['model'] == model_name) & (results_df['channels'] == '2-ch (Pz, F3)')]['balanced_accuracy'].values[0]
    bal_4ch = results_df[(results_df['model'] == model_name) & (results_df['channels'] == '4-ch (Pz, F3, F4, O1)')]['balanced_accuracy'].values[0]
    improvement = ((bal_4ch - bal_2ch) / bal_2ch) * 100
    improvements.append(improvement)

colors = ['green' if imp > 0 else 'red' for imp in improvements]
bars = ax.barh(models_list, improvements, color=colors, alpha=0.7, edgecolor='black')
ax.set_xlabel('Improvement (%)', fontweight='bold')
ax.set_title('Performance Gain from Adding 2 Channels', fontweight='bold')
ax.axvline(0, color='black', linewidth=0.8)
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for bar, val in zip(bars, improvements):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2.,
            f'{val:+.1f}%',
            ha='left' if val > 0 else 'right',
            va='center',
            fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '2ch_vs_4ch_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved comparison plot: {OUTPUT_DIR / '2ch_vs_4ch_comparison.png'}")

# Save report
with open(OUTPUT_DIR / '4channel_comparison_report.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("2-CHANNEL vs 4-CHANNEL COMPARISON REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write("Setup:\n")
    f.write("  2-Channel: Pz (parietal), F3 (frontal left) → 18 features\n")
    f.write("  4-Channel: Pz, F3, F4 (frontal right), O1 (occipital) → 36 features\n\n")
    
    f.write("Models tested (with optimized hyperparameters):\n")
    for model_name, model_info in models.items():
        f.write(f"  - {model_name}: {model_info['description']}\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("RESULTS\n")
    f.write("="*80 + "\n\n")
    
    f.write(results_df.to_string(index=False))
    
    f.write("\n\n" + "="*80 + "\n")
    f.write("BEST OVERALL\n")
    f.write("="*80 + "\n\n")
    f.write(f"Model: {best_row['model']}\n")
    f.write(f"Channels: {best_row['channels']}\n")
    f.write(f"Balanced Accuracy: {best_row['balanced_accuracy']:.4f}\n")
    f.write(f"F1-Score: {best_row['f1_score']:.4f}\n")
    f.write(f"ROC AUC: {best_row['roc_auc']:.4f}\n")
    f.write(f"Sensitivity: {best_row['sensitivity']:.4f}\n")
    f.write(f"Specificity: {best_row['specificity']:.4f}\n")

print(f"✓ Saved report: {OUTPUT_DIR / '4channel_comparison_report.txt'}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
