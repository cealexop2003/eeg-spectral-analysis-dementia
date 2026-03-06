"""
Binary Classification: CN vs Dementia (AD+FTD merged)
======================================================

This script tests if binary classification (healthy vs dementia) performs
better than 3-class classification (CN vs AD vs FTD).

Hypothesis: Since AD and FTD are not statistically distinguishable in 
spectral features, merging them into single "Dementia" class should improve
detection accuracy.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score,
                             roc_curve)
import warnings
warnings.filterwarnings('ignore')

# Paths
FEATURES_FILE = Path('results/part2/simple_stats/features.csv')
OUTPUT_DIR = Path('results/part2/binary_classification')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Settings
RANDOM_SEED = 42
CV_FOLDS = 5

print("="*80)
print("BINARY CLASSIFICATION: CN vs DEMENTIA")
print("="*80)

# Load data
print("\n1. Loading data...")
df = pd.read_csv(FEATURES_FILE)

# Create binary labels: CN=0, Dementia (AD+FTD)=1
df['binary_label'] = df['label'].map({'CN': 'CN', 'AD': 'Dementia', 'FTD': 'Dementia'})

print(f"\nOriginal distribution:")
print(df['label'].value_counts())

print(f"\nBinary distribution:")
print(df['binary_label'].value_counts())

# Prepare features and labels
metadata_cols = ['subject_id', 'label', 'binary_label']
feature_cols = [col for col in df.columns if col not in metadata_cols]

X = df[feature_cols].values
y = df['binary_label'].values

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\nLabel encoding:")
for i, label in enumerate(label_encoder.classes_):
    count = (y_encoded == i).sum()
    print(f"  {label} → {i} ({count} subjects, {count/len(y_encoded)*100:.1f}%)")

# Create models
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
    ]),
    'SVM (RBF)': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED))
    ]),
    'K-Nearest Neighbors': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', KNeighborsClassifier(n_neighbors=5))
    ])
}

# Cross-validation
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

print("\n" + "="*80)
print("EVALUATING MODELS (5-Fold Cross-Validation)")
print("="*80)

results = []

for name, model in models.items():
    print(f"\n{name}:")
    print("-" * 60)
    
    # Accuracy
    acc_scores = cross_val_score(model, X, y_encoded, cv=cv, scoring='accuracy', n_jobs=-1)
    
    # Balanced accuracy
    bal_acc_scores = cross_val_score(model, X, y_encoded, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
    
    # F1 score
    f1_scores = cross_val_score(model, X, y_encoded, cv=cv, scoring='f1', n_jobs=-1)
    
    # ROC AUC
    roc_auc_scores = cross_val_score(model, X, y_encoded, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    # Get predictions for confusion matrix
    y_pred = cross_val_predict(model, X, y_encoded, cv=cv)
    cm = confusion_matrix(y_encoded, y_pred)
    
    # Calculate metrics
    acc_mean, acc_std = acc_scores.mean(), acc_scores.std()
    bal_acc_mean, bal_acc_std = bal_acc_scores.mean(), bal_acc_scores.std()
    f1_mean, f1_std = f1_scores.mean(), f1_scores.std()
    roc_auc_mean, roc_auc_std = roc_auc_scores.mean(), roc_auc_scores.std()
    
    print(f"  Accuracy:          {acc_mean:.4f} ± {acc_std:.4f}")
    print(f"  Balanced Accuracy: {bal_acc_mean:.4f} ± {bal_acc_std:.4f}")
    print(f"  F1-Score:          {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"  ROC AUC:           {roc_auc_mean:.4f} ± {roc_auc_std:.4f}")
    
    print(f"\n  Confusion Matrix:")
    print(f"                Predicted")
    print(f"                CN    Dementia")
    print(f"  Actual CN     {cm[0,0]:3d}      {cm[0,1]:3d}")
    print(f"  Actual Dem    {cm[1,0]:3d}      {cm[1,1]:3d}")
    
    # Sensitivity and Specificity
    TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    
    print(f"\n  Sensitivity (Dementia recall): {sensitivity:.4f}")
    print(f"  Specificity (CN recall):       {specificity:.4f}")
    
    results.append({
        'model': name,
        'accuracy': acc_mean,
        'accuracy_std': acc_std,
        'balanced_accuracy': bal_acc_mean,
        'balanced_accuracy_std': bal_acc_std,
        'f1_score': f1_mean,
        'f1_std': f1_std,
        'roc_auc': roc_auc_mean,
        'roc_auc_std': roc_auc_std,
        'sensitivity': sensitivity,
        'specificity': specificity
    })

# Save results
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('balanced_accuracy', ascending=False)
results_df.to_csv(OUTPUT_DIR / 'binary_classification_results.csv', index=False)

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

print("\nRanked by Balanced Accuracy:")
print(results_df[['model', 'balanced_accuracy', 'f1_score', 'roc_auc']].to_string(index=False))

best_model = results_df.iloc[0]
print(f"\n🏆 Best Model: {best_model['model']}")
print(f"   Balanced Accuracy: {best_model['balanced_accuracy']:.4f} ± {best_model['balanced_accuracy_std']:.4f}")
print(f"   F1-Score: {best_model['f1_score']:.4f} ± {best_model['f1_std']:.4f}")
print(f"   ROC AUC: {best_model['roc_auc']:.4f} ± {best_model['roc_auc_std']:.4f}")
print(f"   Sensitivity: {best_model['sensitivity']:.4f}")
print(f"   Specificity: {best_model['specificity']:.4f}")

# Comparison with 3-class results
print("\n" + "="*80)
print("COMPARISON: Binary vs 3-Class Classification")
print("="*80)

print("\n3-Class (CN vs AD vs FTD):")
print("  Best Model: SVM (RBF)")
print("  Balanced Accuracy: 0.4750 ± 0.0600")
print("  F1-Score (macro): 0.4230 ± 0.0625")

print(f"\nBinary (CN vs Dementia):")
print(f"  Best Model: {best_model['model']}")
print(f"  Balanced Accuracy: {best_model['balanced_accuracy']:.4f} ± {best_model['balanced_accuracy_std']:.4f}")
print(f"  F1-Score: {best_model['f1_score']:.4f} ± {best_model['f1_std']:.4f}")

improvement = ((best_model['balanced_accuracy'] - 0.475) / 0.475) * 100
print(f"\n📊 Improvement: {improvement:+.1f}% in balanced accuracy")

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Model comparison
ax = axes[0]
models_list = results_df['model'].values
bal_acc = results_df['balanced_accuracy'].values
bal_acc_std = results_df['balanced_accuracy_std'].values

x_pos = np.arange(len(models_list))
bars = ax.barh(x_pos, bal_acc, xerr=bal_acc_std, capsize=5, 
               color=['green' if i == 0 else 'steelblue' for i in range(len(models_list))],
               alpha=0.7)
ax.set_yticks(x_pos)
ax.set_yticklabels(models_list)
ax.set_xlabel('Balanced Accuracy', fontweight='bold')
ax.set_title('Binary Classification: Model Comparison', fontweight='bold')
ax.axvline(0.5, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
ax.axvline(0.475, color='red', linestyle='--', alpha=0.5, label='3-class best (47.5%)')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

# Plot 2: Binary vs 3-class
ax = axes[1]
categories = ['3-Class\n(CN/AD/FTD)', 'Binary\n(CN/Dementia)']
values = [0.475, best_model['balanced_accuracy']]
stds = [0.060, best_model['balanced_accuracy_std']]
colors = ['coral', 'limegreen']

bars = ax.bar(categories, values, yerr=stds, capsize=10, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Balanced Accuracy', fontweight='bold')
ax.set_title('Binary vs 3-Class Classification', fontweight='bold')
ax.set_ylim([0, 1])
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
ax.axhline(0.8, color='green', linestyle='--', alpha=0.5, label='Clinical threshold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val, std in zip(bars, values, stds):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}\n±{std:.3f}',
            ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'binary_vs_3class_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved comparison plot: {OUTPUT_DIR / 'binary_vs_3class_comparison.png'}")

# Summary report
with open(OUTPUT_DIR / 'binary_classification_report.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("BINARY CLASSIFICATION REPORT: CN vs DEMENTIA\n")
    f.write("="*80 + "\n\n")
    
    f.write("Dataset:\n")
    f.write(f"  CN (Healthy): 29 subjects (33.0%)\n")
    f.write(f"  Dementia (AD+FTD): 59 subjects (67.0%)\n")
    f.write(f"  Total: 88 subjects\n")
    f.write(f"  Features: {len(feature_cols)}\n\n")
    
    f.write("Validation: Stratified 5-Fold Cross-Validation\n\n")
    
    f.write("Results:\n")
    f.write("-" * 80 + "\n")
    f.write(results_df.to_string(index=False))
    f.write("\n\n")
    
    f.write("Best Model:\n")
    f.write(f"  {best_model['model']}\n")
    f.write(f"  Balanced Accuracy: {best_model['balanced_accuracy']:.4f} ± {best_model['balanced_accuracy_std']:.4f}\n")
    f.write(f"  F1-Score: {best_model['f1_score']:.4f} ± {best_model['f1_std']:.4f}\n")
    f.write(f"  ROC AUC: {best_model['roc_auc']:.4f} ± {best_model['roc_auc_std']:.4f}\n")
    f.write(f"  Sensitivity: {best_model['sensitivity']:.4f}\n")
    f.write(f"  Specificity: {best_model['specificity']:.4f}\n\n")
    
    f.write("="*80 + "\n")
    f.write("COMPARISON WITH 3-CLASS CLASSIFICATION\n")
    f.write("="*80 + "\n\n")
    f.write("3-Class (CN vs AD vs FTD):\n")
    f.write("  Balanced Accuracy: 0.4750 ± 0.0600\n\n")
    f.write(f"Binary (CN vs Dementia):\n")
    f.write(f"  Balanced Accuracy: {best_model['balanced_accuracy']:.4f} ± {best_model['balanced_accuracy_std']:.4f}\n\n")
    f.write(f"Improvement: {improvement:+.1f}%\n")

print(f"✓ Saved report: {OUTPUT_DIR / 'binary_classification_report.txt'}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
