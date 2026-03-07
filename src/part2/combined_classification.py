"""
Combined Classification: Spectral + Connectivity Features
=========================================================

Combines spectral features (36) with connectivity features (18)
to test if connectivity improves classification performance.

Total features: 54 (36 spectral + 18 connectivity)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Paths
SPECTRAL_FILE = Path('results/part2/simple_stats_4ch/features.csv')
CONNECTIVITY_FILE = Path('results/part2/connectivity_features/connectivity_features.csv')
OUTPUT_DIR = Path('results/part2/combined_features')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
CV_FOLDS = 5

print("="*80)
print("COMBINED SPECTRAL + CONNECTIVITY CLASSIFICATION")
print("="*80)

# Load both feature sets
df_spectral = pd.read_csv(SPECTRAL_FILE)
df_connectivity = pd.read_csv(CONNECTIVITY_FILE)

print(f"\nSpectral features: {df_spectral.shape}")
print(f"Connectivity features: {df_connectivity.shape}")

# Merge on subject_id
df_combined = df_spectral.merge(df_connectivity, on=['subject_id', 'label'], how='inner')

print(f"Combined features: {df_combined.shape}")
print(f"\n  CN:  {(df_combined['label'] == 'CN').sum()}")
print(f"  AD:  {(df_combined['label'] == 'AD').sum()}")
print(f"  FTD: {(df_combined['label'] == 'FTD').sum()}")

# Feature columns
metadata_cols = ['subject_id', 'label']
all_features = [col for col in df_combined.columns if col not in metadata_cols]
spectral_features = [col for col in df_spectral.columns if col not in metadata_cols]
connectivity_features = [col for col in df_connectivity.columns if col not in metadata_cols]

print(f"\nFeature breakdown:")
print(f"  Spectral: {len(spectral_features)}")
print(f"  Connectivity: {len(connectivity_features)}")
print(f"  Total: {len(all_features)}")

# Save combined features
df_combined.to_csv(OUTPUT_DIR / 'combined_features.csv', index=False)
print(f"\n✓ Saved: {OUTPUT_DIR / 'combined_features.csv'}")

# Prepare data for binary classification
df_combined['binary_label'] = df_combined['label'].map({'CN': 0, 'AD': 1, 'FTD': 1})

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

print("\n" + "="*80)
print("BINARY CLASSIFICATION: CN vs DEMENTIA")
print("="*80)

# Test three feature sets
feature_sets = {
    'Spectral Only (36 features)': spectral_features,
    'Connectivity Only (18 features)': connectivity_features,
    'Combined (54 features)': all_features
}

results_binary = []

for feature_set_name, features in feature_sets.items():
    print(f"\n{feature_set_name}:")
    print("-" * 60)
    
    X = df_combined[features].values
    y = df_combined['binary_label'].values
    
    # Models
    models = {
        'Logistic Regression': LogisticRegression(C=0.01, class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED),
        'SVM (RBF)': SVC(kernel='rbf', C=1, gamma=0.01, class_weight='balanced', probability=True, random_state=RANDOM_SEED),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=3, class_weight='balanced_subsample', random_state=RANDOM_SEED)
    }
    
    for model_name, model in models.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        # Cross-validation
        bal_acc = cross_val_score(pipeline, X, y, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
        f1 = cross_val_score(pipeline, X, y, cv=cv, scoring='f1', n_jobs=-1)
        roc_auc = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
        
        # Predictions for confusion matrix
        y_pred = cross_val_predict(pipeline, X, y, cv=cv)
        cm = confusion_matrix(y, y_pred)
        TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        sens = TP / (TP + FN)
        spec = TN / (TN + FP)
        
        print(f"\n  {model_name}:")
        print(f"    Balanced Accuracy: {bal_acc.mean():.4f} ± {bal_acc.std():.4f}")
        print(f"    F1-Score:          {f1.mean():.4f} ± {f1.std():.4f}")
        print(f"    ROC AUC:           {roc_auc.mean():.4f} ± {roc_auc.std():.4f}")
        print(f"    Sensitivity:       {sens:.4f}")
        print(f"    Specificity:       {spec:.4f}")
        
        results_binary.append({
            'feature_set': feature_set_name,
            'model': model_name,
            'balanced_accuracy': bal_acc.mean(),
            'bal_acc_std': bal_acc.std(),
            'f1_score': f1.mean(),
            'roc_auc': roc_auc.mean(),
            'sensitivity': sens,
            'specificity': spec
        })

# Binary results DataFrame
df_binary = pd.DataFrame(results_binary)
df_binary.to_csv(OUTPUT_DIR / 'binary_classification_results.csv', index=False)

print("\n" + "="*80)
print("3-CLASS CLASSIFICATION: CN vs AD vs FTD")
print("="*80)

# 3-class labels
label_map = {'CN': 0, 'AD': 1, 'FTD': 2}
df_combined['label_int'] = df_combined['label'].map(label_map)
y_3class = df_combined['label_int'].values

results_3class = []

for feature_set_name, features in feature_sets.items():
    print(f"\n{feature_set_name}:")
    print("-" * 60)
    
    X = df_combined[features].values
    
    models = {
        'Logistic Regression': LogisticRegression(C=1, class_weight='balanced', max_iter=1000, random_state=RANDOM_SEED),
        'SVM (RBF)': SVC(kernel='rbf', C=1, gamma='scale', class_weight='balanced', random_state=RANDOM_SEED),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced_subsample', random_state=RANDOM_SEED)
    }
    
    for model_name, model in models.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model)
        ])
        
        # Cross-validation
        y_pred = cross_val_predict(pipeline, X, y_3class, cv=cv)
        bal_acc = balanced_accuracy_score(y_3class, y_pred)
        f1 = f1_score(y_3class, y_pred, average='macro')
        
        # Per-class recall
        cm = confusion_matrix(y_3class, y_pred)
        recall_CN = cm[0,0] / cm[0,:].sum()
        recall_AD = cm[1,1] / cm[1,:].sum()
        recall_FTD = cm[2,2] / cm[2,:].sum()
        
        print(f"\n  {model_name}:")
        print(f"    Balanced Accuracy: {bal_acc:.4f}")
        print(f"    F1-Score (macro):  {f1:.4f}")
        print(f"    Recall CN:  {recall_CN:.4f}")
        print(f"    Recall AD:  {recall_AD:.4f}")
        print(f"    Recall FTD: {recall_FTD:.4f}")
        
        results_3class.append({
            'feature_set': feature_set_name,
            'model': model_name,
            'balanced_accuracy': bal_acc,
            'f1_macro': f1,
            'recall_CN': recall_CN,
            'recall_AD': recall_AD,
            'recall_FTD': recall_FTD
        })

# 3-class results DataFrame
df_3class = pd.DataFrame(results_3class)
df_3class.to_csv(OUTPUT_DIR / '3class_classification_results.csv', index=False)

# Summary comparison
print("\n" + "="*80)
print("SUMMARY: SPECTRAL vs CONNECTIVITY vs COMBINED")
print("="*80)

print("\nBINARY CLASSIFICATION (Best per feature set):")
for feature_set in feature_sets.keys():
    subset = df_binary[df_binary['feature_set'] == feature_set]
    best_row = subset.loc[subset['balanced_accuracy'].idxmax()]
    print(f"\n  {feature_set}:")
    print(f"    Best Model: {best_row['model']}")
    print(f"    Balanced Accuracy: {best_row['balanced_accuracy']:.4f}")
    print(f"    F1-Score: {best_row['f1_score']:.4f}")
    print(f"    ROC AUC: {best_row['roc_auc']:.4f}")

print("\n3-CLASS CLASSIFICATION (Best per feature set):")
for feature_set in feature_sets.keys():
    subset = df_3class[df_3class['feature_set'] == feature_set]
    best_row = subset.loc[subset['balanced_accuracy'].idxmax()]
    print(f"\n  {feature_set}:")
    print(f"    Best Model: {best_row['model']}")
    print(f"    Balanced Accuracy: {best_row['balanced_accuracy']:.4f}")
    print(f"    FTD Recall: {best_row['recall_FTD']:.4f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Binary classification comparison
ax = axes[0]
for model_name in ['Logistic Regression', 'SVM (RBF)', 'Random Forest']:
    accuracies = []
    for feature_set in feature_sets.keys():
        acc = df_binary[(df_binary['feature_set'] == feature_set) & (df_binary['model'] == model_name)]['balanced_accuracy'].values[0]
        accuracies.append(acc)
    
    x = np.arange(len(feature_sets))
    ax.plot(x, accuracies, marker='o', label=model_name, linewidth=2, markersize=8)

ax.set_xticks(x)
ax.set_xticklabels(['Spectral\n(36)', 'Connectivity\n(18)', 'Combined\n(54)'], fontsize=10)
ax.set_ylabel('Balanced Accuracy', fontweight='bold')
ax.set_title('Binary Classification (CN vs Dementia)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0.5, 0.85])

# 3-class comparison
ax = axes[1]
for model_name in ['Logistic Regression', 'SVM (RBF)', 'Random Forest']:
    accuracies = []
    for feature_set in feature_sets.keys():
        acc = df_3class[(df_3class['feature_set'] == feature_set) & (df_3class['model'] == model_name)]['balanced_accuracy'].values[0]
        accuracies.append(acc)
    
    ax.plot(x, accuracies, marker='o', label=model_name, linewidth=2, markersize=8)

ax.set_xticks(x)
ax.set_xticklabels(['Spectral\n(36)', 'Connectivity\n(18)', 'Combined\n(54)'], fontsize=10)
ax.set_ylabel('Balanced Accuracy', fontweight='bold')
ax.set_title('3-Class Classification (CN/AD/FTD)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(0.333, color='red', linestyle='--', alpha=0.5, label='Chance')
ax.set_ylim([0.3, 0.6])

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'feature_set_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {OUTPUT_DIR / 'feature_set_comparison.png'}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
