"""
Optimized Binary Classification: CN vs Dementia
================================================

Improvements attempted:
1. Feature selection (using FDR-significant features from advanced stats)
2. Hyperparameter tuning (GridSearchCV)
3. Class weighting (balance CN vs Dementia)
4. Ensemble methods (Voting classifier)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             confusion_matrix, classification_report, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

# Paths
FEATURES_FILE = Path('results/part2/simple_stats/features.csv')
OUTPUT_DIR = Path('results/part2/optimized_binary')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
CV_FOLDS = 5

print("="*80)
print("OPTIMIZED BINARY CLASSIFICATION: CN vs DEMENTIA")
print("="*80)

# Load data
df = pd.read_csv(FEATURES_FILE)
df['binary_label'] = df['label'].map({'CN': 0, 'AD': 1, 'FTD': 1})

# Feature columns
metadata_cols = ['subject_id', 'label']
all_feature_cols = [col for col in df.columns if col not in metadata_cols + ['binary_label']]

print(f"\nTotal features available: {len(all_feature_cols)}")

# Define feature sets to test
feature_sets = {}

# 1. All features (baseline)
feature_sets['All Features (18)'] = all_feature_cols

# 2. Top 7 FDR-significant features (from advanced stats)
feature_sets['Top 7 FDR-significant'] = [
    'Pz_theta_alpha',      # p_FDR=0.0037
    'Pz_centroid_4_15',    # p_FDR=0.0073
    'Pz_alpha_rel',        # p_FDR=0.013
    'Pz_slowing_ratio',    # p_FDR=0.013
    'F3_theta_alpha',      # p_FDR=0.014
    'Pz_delta_alpha',      # p_FDR=0.014
    'F3_theta_rel'         # p_FDR=0.023
]

# 3. Top 5 features
feature_sets['Top 5 Features'] = [
    'Pz_theta_alpha',
    'Pz_centroid_4_15',
    'Pz_alpha_rel',
    'Pz_slowing_ratio',
    'F3_theta_alpha'
]

# 4. Top 3 features
feature_sets['Top 3 Features'] = [
    'Pz_theta_alpha',
    'Pz_centroid_4_15',
    'Pz_alpha_rel'
]

# 5. Pz channel only (9 features)
feature_sets['Pz Channel Only'] = [col for col in all_feature_cols if col.startswith('Pz_')]

# 6. F3 channel only (9 features)
feature_sets['F3 Channel Only'] = [col for col in all_feature_cols if col.startswith('F3_')]

# 7. Ratio features only
feature_sets['Ratio Features'] = [
    'Pz_theta_alpha', 'Pz_delta_alpha', 'Pz_slowing_ratio',
    'F3_theta_alpha', 'F3_delta_alpha', 'F3_slowing_ratio'
]

print("\nFeature sets to evaluate:")
for name, features in feature_sets.items():
    print(f"  {name}: {len(features)} features")

# Cross-validation
cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

print("\n" + "="*80)
print("STEP 1: FEATURE SELECTION COMPARISON")
print("="*80)

feature_results = []

for set_name, feature_list in feature_sets.items():
    print(f"\n{set_name} ({len(feature_list)} features):")
    print("-" * 60)
    
    X = df[feature_list].values
    y = df['binary_label'].values
    
    # Test with default SVM (was best in baseline)
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED))
    ])
    
    bal_acc_scores = cross_val_score(model, X, y, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
    roc_auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    bal_acc_mean = bal_acc_scores.mean()
    
    print(f"  Balanced Accuracy: {bal_acc_mean:.4f} ± {bal_acc_scores.std():.4f}")
    print(f"  F1-Score:          {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
    print(f"  ROC AUC:           {roc_auc_scores.mean():.4f} ± {roc_auc_scores.std():.4f}")
    
    feature_results.append({
        'feature_set': set_name,
        'n_features': len(feature_list),
        'balanced_accuracy': bal_acc_mean,
        'bal_acc_std': bal_acc_scores.std(),
        'f1_score': f1_scores.mean(),
        'roc_auc': roc_auc_scores.mean()
    })

# Find best feature set
feature_results_df = pd.DataFrame(feature_results).sort_values('balanced_accuracy', ascending=False)
best_feature_set_name = feature_results_df.iloc[0]['feature_set']
best_features = feature_sets[best_feature_set_name]

print("\n" + "="*80)
print("BEST FEATURE SET:")
print(f"  {best_feature_set_name}")
print(f"  Balanced Accuracy: {feature_results_df.iloc[0]['balanced_accuracy']:.4f}")
print(f"  Features: {best_features}")
print("="*80)

# Prepare data with best features
X_best = df[best_features].values
y = df['binary_label'].values

print("\n" + "="*80)
print("STEP 2: HYPERPARAMETER TUNING")
print("="*80)

# Define parameter grids
param_grids = {
    'SVM': {
        'classifier__C': [0.1, 1, 10, 100],
        'classifier__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'classifier__class_weight': [None, 'balanced']
    },
    'Logistic Regression': {
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l2'],
        'classifier__class_weight': [None, 'balanced']
    },
    'Random Forest': {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [3, 5, 10, None],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__class_weight': [None, 'balanced', 'balanced_subsample']
    }
}

tuning_results = []

for model_name, param_grid in param_grids.items():
    print(f"\n{model_name}:")
    print("-" * 60)
    
    # Create base pipeline
    if model_name == 'SVM':
        base_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='rbf', probability=True, random_state=RANDOM_SEED))
        ])
    elif model_name == 'Logistic Regression':
        base_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=RANDOM_SEED))
        ])
    else:  # Random Forest
        base_model = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=RANDOM_SEED))
        ])
    
    # Grid search
    grid_search = GridSearchCV(
        base_model, 
        param_grid, 
        cv=cv, 
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_best, y)
    
    print(f"  Best parameters: {grid_search.best_params_}")
    print(f"  Best CV score: {grid_search.best_score_:.4f}")
    
    # Evaluate best model
    best_model = grid_search.best_estimator_
    
    bal_acc_scores = cross_val_score(best_model, X_best, y, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
    f1_scores = cross_val_score(best_model, X_best, y, cv=cv, scoring='f1', n_jobs=-1)
    roc_auc_scores = cross_val_score(best_model, X_best, y, cv=cv, scoring='roc_auc', n_jobs=-1)
    
    y_pred = cross_val_predict(best_model, X_best, y, cv=cv)
    cm = confusion_matrix(y, y_pred)
    
    TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    
    print(f"  Balanced Accuracy: {bal_acc_scores.mean():.4f} ± {bal_acc_scores.std():.4f}")
    print(f"  F1-Score:          {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
    print(f"  ROC AUC:           {roc_auc_scores.mean():.4f} ± {roc_auc_scores.std():.4f}")
    print(f"  Sensitivity:       {sensitivity:.4f}")
    print(f"  Specificity:       {specificity:.4f}")
    
    tuning_results.append({
        'model': model_name,
        'balanced_accuracy': bal_acc_scores.mean(),
        'bal_acc_std': bal_acc_scores.std(),
        'f1_score': f1_scores.mean(),
        'roc_auc': roc_auc_scores.mean(),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'best_params': grid_search.best_params_
    })

# Best tuned model
tuning_results_df = pd.DataFrame(tuning_results).sort_values('balanced_accuracy', ascending=False)
best_tuned = tuning_results_df.iloc[0]

print("\n" + "="*80)
print("STEP 3: ENSEMBLE METHOD")
print("="*80)

# Create ensemble with best hyperparameters
print("\nCreating Voting Classifier with optimized models...")

# Get best models from tuning
estimators = []
for _, row in tuning_results_df.head(3).iterrows():
    model_name = row['model']
    params = row['best_params']
    
    if model_name == 'SVM':
        clf = SVC(
            kernel='rbf',
            C=params.get('classifier__C', 1.0),
            gamma=params.get('classifier__gamma', 'scale'),
            probability=True,
            random_state=RANDOM_SEED
        )
    elif model_name == 'Logistic Regression':
        clf = LogisticRegression(
            C=params.get('classifier__C', 1.0),
            max_iter=1000,
            random_state=RANDOM_SEED
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=params.get('classifier__n_estimators', 100),
            max_depth=params.get('classifier__max_depth', None),
            random_state=RANDOM_SEED
        )
    
    estimators.append((model_name.lower().replace(' ', '_'), clf))

ensemble = Pipeline([
    ('scaler', StandardScaler()),
    ('voting', VotingClassifier(estimators=estimators, voting='soft'))
])

# Evaluate ensemble
bal_acc_scores = cross_val_score(ensemble, X_best, y, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
f1_scores = cross_val_score(ensemble, X_best, y, cv=cv, scoring='f1', n_jobs=-1)
roc_auc_scores = cross_val_score(ensemble, X_best, y, cv=cv, scoring='roc_auc', n_jobs=-1)

y_pred = cross_val_predict(ensemble, X_best, y, cv=cv)
cm = confusion_matrix(y, y_pred)

TN, FP, FN, TP = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
ensemble_sensitivity = TP / (TP + FN)
ensemble_specificity = TN / (TN + FP)

print(f"\nEnsemble Performance:")
print(f"  Balanced Accuracy: {bal_acc_scores.mean():.4f} ± {bal_acc_scores.std():.4f}")
print(f"  F1-Score:          {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
print(f"  ROC AUC:           {roc_auc_scores.mean():.4f} ± {roc_auc_scores.std():.4f}")
print(f"  Sensitivity:       {ensemble_sensitivity:.4f}")
print(f"  Specificity:       {ensemble_specificity:.4f}")

print("\n  Confusion Matrix:")
print(f"                CN    Dementia")
print(f"  Actual CN     {cm[0,0]:3d}      {cm[0,1]:3d}")
print(f"  Actual Dem    {cm[1,0]:3d}      {cm[1,1]:3d}")

# Final comparison
print("\n" + "="*80)
print("FINAL COMPARISON")
print("="*80)

comparison = pd.DataFrame([
    {
        'Approach': 'Baseline (all features, default params)',
        'Balanced Acc': 0.6952,
        'F1': 0.8340,
        'ROC AUC': 0.7841
    },
    {
        'Approach': f'Best features ({best_feature_set_name})',
        'Balanced Acc': feature_results_df.iloc[0]['balanced_accuracy'],
        'F1': feature_results_df.iloc[0]['f1_score'],
        'ROC AUC': feature_results_df.iloc[0]['roc_auc']
    },
    {
        'Approach': f'Best tuned ({best_tuned["model"]})',
        'Balanced Acc': best_tuned['balanced_accuracy'],
        'F1': best_tuned['f1_score'],
        'ROC AUC': best_tuned['roc_auc']
    },
    {
        'Approach': 'Ensemble (voting)',
        'Balanced Acc': bal_acc_scores.mean(),
        'F1': f1_scores.mean(),
        'ROC AUC': roc_auc_scores.mean()
    }
])

print("\n" + comparison.to_string(index=False))

# Save results
comparison.to_csv(OUTPUT_DIR / 'optimization_comparison.csv', index=False)
feature_results_df.to_csv(OUTPUT_DIR / 'feature_selection_results.csv', index=False)
tuning_results_df.to_csv(OUTPUT_DIR / 'hyperparameter_tuning_results.csv', index=False)

# Find overall best
best_overall_idx = comparison['Balanced Acc'].idxmax()
best_overall = comparison.iloc[best_overall_idx]

print("\n" + "="*80)
print("🏆 BEST OVERALL RESULT:")
print(f"   {best_overall['Approach']}")
print(f"   Balanced Accuracy: {best_overall['Balanced Acc']:.4f}")
print(f"   F1-Score: {best_overall['F1']:.4f}")
print(f"   ROC AUC: {best_overall['ROC AUC']:.4f}")
print("="*80)

# Improvement analysis
baseline_bal_acc = 0.6952
best_bal_acc = best_overall['Balanced Acc']
improvement = ((best_bal_acc - baseline_bal_acc) / baseline_bal_acc) * 100

print(f"\n📊 Improvement over baseline: {improvement:+.1f}%")

# Realistic assessment
print("\n" + "="*80)
print("REALISTIC ASSESSMENT")
print("="*80)

print(f"""
Current Best: {best_bal_acc:.1%} balanced accuracy

Target: 90% balanced accuracy
Gap: {(0.90 - best_bal_acc):.1%}

Why 90% is difficult with this dataset:
1. Low signal-to-noise ratio (PCA showed 51% variance is noise)
2. Small sample size (N=88, especially CN=29)
3. High individual variability in EEG
4. Limited features (spectral only, 2 channels)
5. Heterogeneous dementia pathology

To reach 90%, would need:
- More subjects (N>200)
- All 19 EEG channels (spatial features)
- Connectivity features (coherence, PLV)
- Temporal features (entropy, complexity)
- Multimodal data (EEG + MRI + cognitive scores)
- Deep learning on raw EEG signals

Current performance ({best_bal_acc:.1%}) is reasonable for:
- Screening tool (not diagnostic)
- Preliminary risk assessment
- Triage for further testing
""")

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
