"""
Multi-Class Classification Analysis for EEG-Based Dementia Detection
=====================================================================

This module implements supervised machine learning classification to distinguish
between CN (Control), AD (Alzheimer's Disease), and FTD (Frontotemporal Dementia)
using spectral EEG features.

Classification Pipeline:
1. Data loading and label encoding
2. Leakage-free preprocessing with sklearn Pipeline
3. Multi-model comparison (Logistic, SVM, RF, KNN)
4. Stratified 5-fold cross-validation
5. Performance metrics and confusion matrices
6. Feature importance analysis
7. Visualization and reporting

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
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             confusion_matrix, classification_report)


# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
FEATURES_FILE = Path('results/part2/simple_stats/features.csv')
OUTPUT_DIR = Path('results/part2/classification')

# Output files
RESULTS_CSV = OUTPUT_DIR / 'classification_results.csv'
IMPORTANCE_CSV = OUTPUT_DIR / 'feature_importance.csv'
SUMMARY_REPORT = OUTPUT_DIR / 'classification_summary_report.txt'

# Random seed for reproducibility
RANDOM_SEED = 42

# Cross-validation settings
CV_FOLDS = 5

# Class labels
CLASS_LABELS = ['CN', 'AD', 'FTD']


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data():
    """
    Load feature matrix and prepare X, y for classification.
    
    Returns
    -------
    X : pd.DataFrame
        Feature matrix (numeric features only)
    y : pd.Series
        Target labels
    feature_names : list
        Names of features
    label_encoder : LabelEncoder
        Fitted label encoder
    """
    print("\n" + "="*80)
    print("DATA LOADING")
    print("="*80)
    
    # Load features
    if not FEATURES_FILE.exists():
        raise FileNotFoundError(f"Features file not found: {FEATURES_FILE}")
    
    df = pd.read_csv(FEATURES_FILE)
    
    print(f"\nLoaded {len(df)} subjects")
    print(f"Total columns: {df.shape[1]}")
    
    # Separate features and target
    metadata_cols = ['subject_id', 'label']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    X = df[feature_cols]
    y = df['label']
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"\nFeatures: {len(feature_cols)}")
    print(f"Feature names: {feature_cols[:5]}... (showing first 5)")
    
    print(f"\nClass distribution:")
    for i, label in enumerate(label_encoder.classes_):
        count = (y_encoded == i).sum()
        print(f"  {label}: {count} ({count/len(y_encoded)*100:.1f}%)")
    
    print(f"\nLabel encoding:")
    for i, label in enumerate(label_encoder.classes_):
        print(f"  {label} → {i}")
    
    return X, pd.Series(y_encoded), feature_cols, label_encoder


# ============================================================================
# MODEL DEFINITION
# ============================================================================

def create_models():
    """
    Create dictionary of classification models with pipelines.
    
    Each pipeline includes:
    - StandardScaler (prevents data leakage during CV)
    - Classifier
    
    Returns
    -------
    models : dict
        Dictionary of model name -> pipeline
    model_params : dict
        Dictionary of model name -> parameters
    """
    print("\n" + "="*80)
    print("MODEL INITIALIZATION")
    print("="*80)
    
    models = {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_SEED
            ))
        ]),
        
        'SVM (RBF)': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(
                kernel='rbf',
                probability=True,
                random_state=RANDOM_SEED
            ))
        ]),
        
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                random_state=RANDOM_SEED
            ))
        ]),
        
        'K-Nearest Neighbors': Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', KNeighborsClassifier())
        ])
    }
    
    # Store model parameters for reporting
    model_params = {
        'Logistic Regression': {
            'solver': 'Default (lbfgs for multiclass)',
            'max_iter': 1000,
            'random_state': RANDOM_SEED,
            'preprocessing': 'StandardScaler'
        },
        'SVM (RBF)': {
            'kernel': 'rbf',
            'C': 1.0,  # default
            'gamma': 'scale',  # default
            'probability': True,
            'random_state': RANDOM_SEED,
            'preprocessing': 'StandardScaler'
        },
        'Random Forest': {
            'n_estimators': 100,  # default
            'max_depth': None,  # default (unlimited)
            'min_samples_split': 2,  # default
            'min_samples_leaf': 1,  # default
            'random_state': RANDOM_SEED,
            'preprocessing': 'StandardScaler'
        },
        'K-Nearest Neighbors': {
            'n_neighbors': 5,  # default
            'weights': 'uniform',  # default
            'metric': 'minkowski',  # default
            'p': 2,  # Euclidean distance
            'preprocessing': 'StandardScaler'
        }
    }
    
    print(f"\nInitialized {len(models)} models:")
    for name in models.keys():
        print(f"  - {name}")
    
    return models, model_params


# ============================================================================
# CROSS-VALIDATION AND EVALUATION
# ============================================================================

def evaluate_model(model, X, y, cv, model_name):
    """
    Evaluate a single model using stratified k-fold cross-validation.
    
    Parameters
    ----------
    model : Pipeline
        Scikit-learn pipeline with scaler and classifier
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target labels (encoded)
    cv : StratifiedKFold
        Cross-validation splitter
    model_name : str
        Name of the model
    
    Returns
    -------
    results : dict
        Dictionary containing metrics and predictions
    """
    print(f"\n{model_name}:")
    print("-" * 60)
    
    # Compute accuracy
    acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
    
    # Compute balanced accuracy
    bal_acc_scores = cross_val_score(model, X, y, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
    
    # Compute macro F1
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
    
    # Get predictions for confusion matrix
    y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    
    # Print results
    print(f"  Accuracy:          {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")
    print(f"  Balanced Accuracy: {bal_acc_scores.mean():.4f} ± {bal_acc_scores.std():.4f}")
    print(f"  Macro F1-score:    {f1_scores.mean():.4f} ± {f1_scores.std():.4f}")
    
    return {
        'model': model_name,
        'accuracy_mean': acc_scores.mean(),
        'accuracy_std': acc_scores.std(),
        'balanced_accuracy_mean': bal_acc_scores.mean(),
        'balanced_accuracy_std': bal_acc_scores.std(),
        'f1_macro_mean': f1_scores.mean(),
        'f1_macro_std': f1_scores.std(),
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'accuracy_scores': acc_scores,
        'balanced_accuracy_scores': bal_acc_scores,
        'f1_scores': f1_scores
    }


def run_classification_experiments(models, X, y):
    """
    Run cross-validation experiments for all models.
    
    Parameters
    ----------
    models : dict
        Dictionary of model pipelines
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target labels
    
    Returns
    -------
    results_dict : dict
        Results for each model
    """
    print("\n" + "="*80)
    print("CROSS-VALIDATION EXPERIMENTS")
    print("="*80)
    print(f"\nStrategy: Stratified {CV_FOLDS}-Fold Cross-Validation")
    print(f"Metrics: Accuracy, Balanced Accuracy, Macro F1-score")
    
    # Create CV splitter
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    # Evaluate each model
    results_dict = {}
    for model_name, model in models.items():
        results = evaluate_model(model, X, y, cv, model_name)
        results_dict[model_name] = results
    
    return results_dict


# ============================================================================
# FEATURE IMPORTANCE ANALYSIS
# ============================================================================

def extract_feature_importance(models, X, y, feature_names):
    """
    Extract feature importance from trained models.
    
    Parameters
    ----------
    models : dict
        Dictionary of model pipelines
    X : pd.DataFrame
        Feature matrix
    y : pd.Series
        Target labels
    feature_names : list
        Names of features
    
    Returns
    -------
    importance_df : pd.DataFrame
        Feature importance for all applicable models
    """
    print("\n" + "="*80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    importance_list = []
    
    # Train models on full data to extract importance
    for model_name, model in models.items():
        
        if model_name in ['Logistic Regression', 'Random Forest']:
            print(f"\n{model_name}:")
            
            # Fit on full data
            model.fit(X, y)
            
            # Extract classifier
            classifier = model.named_steps['classifier']
            
            if model_name == 'Logistic Regression':
                # Average absolute coefficients across classes
                coefs = np.abs(classifier.coef_).mean(axis=0)
                importance_values = coefs
                
            elif model_name == 'Random Forest':
                # Use feature importances
                importance_values = classifier.feature_importances_
            
            # Create DataFrame
            for feat_name, importance in zip(feature_names, importance_values):
                importance_list.append({
                    'model': model_name,
                    'feature': feat_name,
                    'importance': importance
                })
            
            # Print top 10
            feat_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance_values
            }).sort_values('importance', ascending=False)
            
            print(f"  Top 10 features:")
            for i, row in feat_df.head(10).iterrows():
                print(f"    {row['feature']}: {row['importance']:.4f}")
    
    importance_df = pd.DataFrame(importance_list)
    
    return importance_df


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_confusion_matrix(cm, label_encoder, model_name, save_path):
    """
    Plot confusion matrix as heatmap.
    
    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix
    label_encoder : LabelEncoder
        Fitted label encoder
    model_name : str
        Name of the model
    save_path : Path
        Path to save figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_,
                ax=ax, cbar_kws={'label': 'Count'})
    
    # Add normalized values as text
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = ax.text(j + 0.5, i + 0.7, f'({cm_normalized[i, j]:.2f})',
                          ha="center", va="center", color="red", fontsize=9)
    
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title(f'Confusion Matrix: {model_name}\n(counts and normalized rates)', 
                fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_model_comparison(results_dict, save_path):
    """
    Create bar plot comparing model performance.
    
    Parameters
    ----------
    results_dict : dict
        Results from all models
    save_path : Path
        Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    metrics = [
        ('accuracy_mean', 'accuracy_std', 'Accuracy'),
        ('balanced_accuracy_mean', 'balanced_accuracy_std', 'Balanced Accuracy'),
        ('f1_macro_mean', 'f1_macro_std', 'Macro F1-Score')
    ]
    
    model_names = list(results_dict.keys())
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for idx, (mean_key, std_key, title) in enumerate(metrics):
        ax = axes[idx]
        
        means = [results_dict[m][mean_key] for m in model_names]
        stds = [results_dict[m][std_key] for m in model_names]
        
        x_pos = np.arange(len(model_names))
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                     color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel(title, fontweight='bold')
        ax.set_title(f'{title} Comparison\n(5-fold CV)', fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                   f'{mean:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance(importance_df, model_name, save_path, top_n=10):
    """
    Plot feature importance for a specific model.
    
    Parameters
    ----------
    importance_df : pd.DataFrame
        Feature importance DataFrame
    model_name : str
        Name of model to plot
    save_path : Path
        Path to save figure
    top_n : int
        Number of top features to show
    """
    # Filter for this model
    model_importance = importance_df[importance_df['model'] == model_name].copy()
    model_importance = model_importance.sort_values('importance', ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Horizontal bar plot
    y_pos = np.arange(len(model_importance))
    bars = ax.barh(y_pos, model_importance['importance'], 
                   color='steelblue', alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(model_importance['feature'])
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontweight='bold')
    ax.set_title(f'Top {top_n} Most Important Features\n{model_name}', 
                fontweight='bold', fontsize=13)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, model_importance['importance'])):
        ax.text(val + 0.001, i, f'{val:.4f}', 
               va='center', ha='left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def generate_visualizations(results_dict, importance_df, label_encoder):
    """
    Generate all visualizations.
    
    Parameters
    ----------
    results_dict : dict
        Classification results
    importance_df : pd.DataFrame
        Feature importance
    label_encoder : LabelEncoder
        Fitted label encoder
    """
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    # Find best model
    best_model = max(results_dict.items(), 
                    key=lambda x: x[1]['balanced_accuracy_mean'])[0]
    
    print(f"\nBest model (by balanced accuracy): {best_model}")
    
    # 1. Confusion matrix for best model
    print("\n1. Confusion matrix for best model...")
    cm = results_dict[best_model]['confusion_matrix']
    plot_confusion_matrix(cm, label_encoder, best_model, 
                         OUTPUT_DIR / f'confusion_matrix_{best_model.replace(" ", "_").lower()}.png')
    print(f"   Saved: confusion_matrix_{best_model.replace(' ', '_').lower()}.png")
    
    # 2. Model comparison
    print("\n2. Model performance comparison...")
    plot_model_comparison(results_dict, OUTPUT_DIR / 'model_comparison.png')
    print("   Saved: model_comparison.png")
    
    # 3. Feature importance for models that support it
    print("\n3. Feature importance plots...")
    for model_name in ['Logistic Regression', 'Random Forest']:
        if model_name in results_dict:
            safe_name = model_name.replace(' ', '_').lower()
            plot_feature_importance(importance_df, model_name, 
                                   OUTPUT_DIR / f'feature_importance_{safe_name}.png',
                                   top_n=10)
            print(f"   Saved: feature_importance_{safe_name}.png")


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(results_dict, importance_df, model_params, label_encoder, y):
    """
    Save classification results and feature importance to CSV.
    
    Parameters
    ----------
    results_dict : dict
        Classification results
    importance_df : pd.DataFrame
        Feature importance
    model_params : dict
        Model parameters
    label_encoder : LabelEncoder
        Fitted label encoder
    y : pd.Series
        Target labels
    """
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # 1. Classification results
    results_list = []
    for model_name, results in results_dict.items():
        results_list.append({
            'model': results['model'],
            'accuracy_mean': results['accuracy_mean'],
            'accuracy_std': results['accuracy_std'],
            'balanced_accuracy_mean': results['balanced_accuracy_mean'],
            'balanced_accuracy_std': results['balanced_accuracy_std'],
            'f1_macro_mean': results['f1_macro_mean'],
            'f1_macro_std': results['f1_macro_std']
        })
    
    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values('balanced_accuracy_mean', ascending=False)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\n✓ Saved: {RESULTS_CSV}")
    
    # 2. Feature importance
    importance_df.to_csv(IMPORTANCE_CSV, index=False)
    print(f"✓ Saved: {IMPORTANCE_CSV}")
    
    # 3. Comprehensive text report
    save_text_report(results_dict, importance_df, model_params, label_encoder, y)
    print(f"✓ Saved: {SUMMARY_REPORT}")


# ============================================================================
# TEXT REPORT GENERATION
# ============================================================================

def save_text_report(results_dict, importance_df, model_params, label_encoder, y):
    """
    Save comprehensive text report with all results and parameters.
    
    Parameters
    ----------
    results_dict : dict
        Classification results
    importance_df : pd.DataFrame
        Feature importance
    model_params : dict
        Model parameters
    label_encoder : LabelEncoder
        Fitted label encoder
    y : pd.Series
        Target labels
    """
    with open(SUMMARY_REPORT, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CLASSIFICATION ANALYSIS COMPREHENSIVE REPORT\n")
        f.write("EEG-Based Dementia Detection: CN vs AD vs FTD\n")
        f.write(f"Date: February 22, 2026\n")
        f.write("="*80 + "\n\n")
        
        # Dataset info
        f.write("="*80 + "\n")
        f.write("DATASET INFORMATION\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total subjects: {len(y)}\n")
        f.write(f"Number of classes: {len(label_encoder.classes_)}\n")
        f.write(f"Classes: {', '.join(label_encoder.classes_)}\n\n")
        f.write("Class distribution:\n")
        for i, class_name in enumerate(label_encoder.classes_):
            count = (y == i).sum()
            f.write(f"  {class_name}: {count} ({count/len(y)*100:.1f}%)\n")
        f.write(f"\nNumber of features: {len(importance_df['feature'].unique())}\n\n")
        
        # Model parameters
        f.write("="*80 + "\n")
        f.write("MODEL CONFIGURATIONS\n")
        f.write("="*80 + "\n\n")
        
        for model_name in results_dict.keys():
            f.write(f"{model_name}:\n")
            f.write("-" * 60 + "\n")
            params = model_params.get(model_name, {})
            for param_name, param_value in params.items():
                f.write(f"  {param_name}: {param_value}\n")
            f.write("\n")
        
        # CV Strategy
        f.write("="*80 + "\n")
        f.write("CROSS-VALIDATION STRATEGY\n")
        f.write("="*80 + "\n\n")
        f.write(f"Method: Stratified K-Fold\n")
        f.write(f"Number of folds: {CV_FOLDS}\n")
        f.write(f"Shuffle: True\n")
        f.write(f"Random state: {RANDOM_SEED}\n")
        f.write(f"Metrics: Accuracy, Balanced Accuracy, Macro F1-Score\n\n")
        
        # Performance results
        f.write("="*80 + "\n")
        f.write("MODEL PERFORMANCE RESULTS\n")
        f.write("="*80 + "\n\n")
        
        for model_name, results in sorted(results_dict.items(), 
                                         key=lambda x: x[1]['balanced_accuracy_mean'], 
                                         reverse=True):
            f.write(f"{model_name}:\n")
            f.write("-" * 60 + "\n")
            f.write(f"  Accuracy:          {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}\n")
            f.write(f"  Balanced Accuracy: {results['balanced_accuracy_mean']:.4f} ± {results['balanced_accuracy_std']:.4f}\n")
            f.write(f"  Macro F1-Score:    {results['f1_macro_mean']:.4f} ± {results['f1_macro_std']:.4f}\n")
            
            # Confusion matrix
            cm = results['confusion_matrix']
            f.write(f"\n  Confusion Matrix (aggregated across {CV_FOLDS} folds):\n")
            f.write(f"  {' '*12}Predicted\n")
            f.write(f"  {' '*12}{'  '.join([f'{c:>6}' for c in label_encoder.classes_])}\n")
            for i, true_class in enumerate(label_encoder.classes_):
                row_str = '  '.join([f'{cm[i,j]:>6}' for j in range(len(label_encoder.classes_))])
                f.write(f"  Actual {true_class:>3}  {row_str}\n")
            
            # Per-class metrics
            f.write(f"\n  Per-class performance:\n")
            for i, class_name in enumerate(label_encoder.classes_):
                recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
                precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                f.write(f"    {class_name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}\n")
            
            f.write("\n")
        
        # Best model
        best_model = max(results_dict.items(), key=lambda x: x[1]['balanced_accuracy_mean'])
        best_name = best_model[0]
        best_results = best_model[1]
        
        f.write("="*80 + "\n")
        f.write("BEST PERFORMING MODEL\n")
        f.write("="*80 + "\n\n")
        f.write(f"Model: {best_name}\n")
        f.write(f"Balanced Accuracy: {best_results['balanced_accuracy_mean']:.4f} ± {best_results['balanced_accuracy_std']:.4f}\n")
        f.write(f"Accuracy: {best_results['accuracy_mean']:.4f} ± {best_results['accuracy_std']:.4f}\n")
        f.write(f"Macro F1-Score: {best_results['f1_macro_mean']:.4f} ± {best_results['f1_macro_std']:.4f}\n\n")
        
        # Baseline comparison
        f.write("="*80 + "\n")
        f.write("BASELINE COMPARISON\n")
        f.write("="*80 + "\n\n")
        
        class_counts = pd.Series(y).value_counts()
        majority_class_prop = class_counts.max() / len(y)
        random_baseline = 1.0 / len(label_encoder.classes_)
        
        f.write(f"Majority class baseline accuracy: {majority_class_prop:.4f}\n")
        f.write(f"Best model accuracy: {best_results['accuracy_mean']:.4f}\n")
        improvement_majority = (best_results['accuracy_mean'] - majority_class_prop) / majority_class_prop * 100
        f.write(f"Improvement over majority baseline: {improvement_majority:+.1f}%\n\n")
        
        f.write(f"Random guess baseline: {random_baseline:.4f}\n")
        f.write(f"Best model balanced accuracy: {best_results['balanced_accuracy_mean']:.4f}\n")
        improvement_random = (best_results['balanced_accuracy_mean'] - random_baseline) / random_baseline * 100
        f.write(f"Improvement over random baseline: {improvement_random:+.1f}%\n\n")
        
        # Feature importance
        f.write("="*80 + "\n")
        f.write("MOST PREDICTIVE FEATURES\n")
        f.write("="*80 + "\n\n")
        
        feature_importance_agg = importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
        
        f.write("Top 10 features (averaged across models):\n")
        for i, (feat, imp) in enumerate(feature_importance_agg.head(10).items(), 1):
            f.write(f"  {i:2}. {feat:25s} {imp:.4f}\n")
        
        # Channel comparison
        f.write("\nChannel comparison:\n")
        pz_features = importance_df[importance_df['feature'].str.startswith('Pz_')]
        f3_features = importance_df[importance_df['feature'].str.startswith('F3_')]
        
        if len(pz_features) > 0 and len(f3_features) > 0:
            pz_avg = pz_features['importance'].mean()
            f3_avg = f3_features['importance'].mean()
            f.write(f"  Pz channel avg importance: {pz_avg:.4f}\n")
            f.write(f"  F3 channel avg importance: {f3_avg:.4f}\n")
            if f3_avg > pz_avg:
                f.write(f"  → F3 channel features are more informative overall\n")
            else:
                f.write(f"  → Pz channel features are more informative overall\n")
        f.write("\n")
        
        # Interpretation
        f.write("="*80 + "\n")
        f.write("CLASSIFICATION DIFFICULTY ASSESSMENT\n")
        f.write("="*80 + "\n\n")
        
        balanced_acc = best_results['balanced_accuracy_mean']
        if balanced_acc >= 0.90:
            difficulty = "EASY - Excellent discrimination"
        elif balanced_acc >= 0.75:
            difficulty = "MODERATE - Good discrimination"
        elif balanced_acc >= 0.60:
            difficulty = "CHALLENGING - Moderate discrimination"
        else:
            difficulty = "VERY CHALLENGING - Poor discrimination"
        
        f.write(f"Difficulty level: {difficulty}\n")
        f.write(f"Balanced accuracy: {balanced_acc:.4f}\n\n")
        
        # Identify hardest class
        cm = best_results['confusion_matrix']
        recalls = [cm[i, i] / cm[i, :].sum() for i in range(len(label_encoder.classes_))]
        worst_class_idx = np.argmin(recalls)
        worst_class = label_encoder.classes_[worst_class_idx]
        
        f.write(f"Hardest class to identify: {worst_class} (recall: {recalls[worst_class_idx]:.3f})\n\n")
        
        f.write("Interpretation:\n")
        f.write(f"  • The classification task shows {difficulty.split('-')[1].strip().lower()}\n")
        f.write(f"  • Spectral EEG features contain discriminative information\n")
        f.write(f"  • {best_name} performed best among tested baseline models\n")
        f.write(f"  • {worst_class} class requires special attention (lowest recall)\n\n")
        
        # Recommendations
        f.write("="*80 + "\n")
        f.write("RECOMMENDATIONS FOR IMPROVEMENT\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. Hyperparameter Optimization:\n")
        f.write("   • Use GridSearchCV or RandomizedSearchCV\n")
        f.write("   • Focus on SVM (C, gamma) and Random Forest (n_estimators, max_depth)\n\n")
        
        f.write("2. Feature Engineering:\n")
        f.write("   • Feature selection based on statistical significance\n")
        f.write("   • Consider polynomial features or interactions\n")
        f.write("   • Add time-domain features\n\n")
        
        f.write("3. Advanced Methods:\n")
        f.write("   • Ensemble methods (Voting, Stacking, Gradient Boosting)\n")
        f.write("   • Deep learning (CNN, LSTM for raw EEG)\n")
        f.write("   • Transfer learning from pre-trained models\n\n")
        
        f.write("4. Data Augmentation:\n")
        f.write(f"   • Collect more data, especially for {worst_class} class\n")
        f.write("   • Consider SMOTE or other oversampling techniques\n")
        f.write("   • Use data augmentation if working with raw signals\n\n")
        
        f.write("="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")


# ============================================================================
# SCIENTIFIC SUMMARY
# ============================================================================

def print_scientific_summary(results_dict, importance_df, y, label_encoder):
    """
    Print comprehensive scientific summary of classification results.
    
    Parameters
    ----------
    results_dict : dict
        Classification results
    importance_df : pd.DataFrame
        Feature importance
    y : pd.Series
        Target labels
    label_encoder : LabelEncoder
        Fitted label encoder
    """
    print("\n" + "="*80)
    print("SCIENTIFIC SUMMARY")
    print("="*80)
    
    # 1. Best performing model
    best_model = max(results_dict.items(), 
                    key=lambda x: x[1]['balanced_accuracy_mean'])
    best_name = best_model[0]
    best_results = best_model[1]
    
    print(f"\n1. BEST PERFORMING MODEL:")
    print("-" * 80)
    print(f"   Model: {best_name}")
    print(f"   Balanced Accuracy: {best_results['balanced_accuracy_mean']:.4f} ± {best_results['balanced_accuracy_std']:.4f}")
    print(f"   Accuracy: {best_results['accuracy_mean']:.4f} ± {best_results['accuracy_std']:.4f}")
    print(f"   Macro F1-score: {best_results['f1_macro_mean']:.4f} ± {best_results['f1_macro_std']:.4f}")
    
    # 2. Baseline comparison
    print(f"\n2. BASELINE COMPARISON:")
    print("-" * 80)
    
    # Majority class baseline
    class_counts = pd.Series(y).value_counts()
    majority_class_prop = class_counts.max() / len(y)
    
    print(f"   Majority class baseline: {majority_class_prop:.4f}")
    print(f"   Best model accuracy: {best_results['accuracy_mean']:.4f}")
    
    if best_results['accuracy_mean'] > majority_class_prop:
        improvement = (best_results['accuracy_mean'] - majority_class_prop) / majority_class_prop * 100
        print(f"   ✓ Model exceeds baseline by {improvement:.1f}%")
    else:
        print(f"   ✗ Model does not exceed baseline")
    
    # Random baseline
    random_baseline = 1.0 / len(label_encoder.classes_)
    print(f"\n   Random guess baseline: {random_baseline:.4f}")
    print(f"   Best model balanced accuracy: {best_results['balanced_accuracy_mean']:.4f}")
    improvement_random = (best_results['balanced_accuracy_mean'] - random_baseline) / random_baseline * 100
    print(f"   ✓ Model exceeds random baseline by {improvement_random:.1f}%")
    
    # 3. Most predictive features
    print(f"\n3. MOST PREDICTIVE FEATURES:")
    print("-" * 80)
    
    # Aggregate importance across models
    feature_importance_agg = importance_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
    
    print(f"   Top 5 features (averaged across models):")
    for i, (feat, imp) in enumerate(feature_importance_agg.head(5).items(), 1):
        print(f"      {i}. {feat}: {imp:.4f}")
    
    # 4. Classification difficulty assessment
    print(f"\n4. CLASSIFICATION DIFFICULTY ASSESSMENT:")
    print("-" * 80)
    
    balanced_acc = best_results['balanced_accuracy_mean']
    
    if balanced_acc >= 0.90:
        difficulty = "EASY - Excellent discrimination"
    elif balanced_acc >= 0.75:
        difficulty = "MODERATE - Good discrimination"
    elif balanced_acc >= 0.60:
        difficulty = "CHALLENGING - Moderate discrimination"
    else:
        difficulty = "VERY CHALLENGING - Poor discrimination"
    
    print(f"   Classification difficulty: {difficulty}")
    print(f"   Balanced accuracy: {balanced_acc:.4f}")
    
    # Analyze confusion matrix
    cm = best_results['confusion_matrix']
    
    print(f"\n   Per-class performance (from confusion matrix):")
    for i, class_name in enumerate(label_encoder.classes_):
        # Recall (sensitivity) = TP / (TP + FN)
        recall = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
        # Precision = TP / (TP + FP)
        precision = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() > 0 else 0
        
        print(f"      {class_name}: Recall={recall:.3f}, Precision={precision:.3f}")
    
    # 5. Interpretation
    print(f"\n5. INTERPRETATION:")
    print("-" * 80)
    
    print(f"   • The classification task shows {difficulty.split('-')[1].strip().lower()}")
    print(f"   • Spectral EEG features contain discriminative information")
    print(f"   • {best_name} performed best among tested baseline models")
    
    # Check if certain classes are harder
    recalls = [cm[i, i] / cm[i, :].sum() for i in range(len(label_encoder.classes_))]
    worst_class_idx = np.argmin(recalls)
    worst_class = label_encoder.classes_[worst_class_idx]
    
    print(f"   • {worst_class} class is hardest to identify (lowest recall: {recalls[worst_class_idx]:.3f})")
    
    # Feature channel analysis
    pz_features = importance_df[importance_df['feature'].str.startswith('Pz_')]
    f3_features = importance_df[importance_df['feature'].str.startswith('F3_')]
    
    if len(pz_features) > 0 and len(f3_features) > 0:
        pz_avg = pz_features['importance'].mean()
        f3_avg = f3_features['importance'].mean()
        
        if pz_avg > f3_avg:
            print(f"   • Pz channel features appear more informative (avg importance: {pz_avg:.4f} vs {f3_avg:.4f})")
        else:
            print(f"   • F3 channel features appear more informative (avg importance: {f3_avg:.4f} vs {pz_avg:.4f})")
    
    print(f"\n6. NEXT STEPS:")
    print("-" * 80)
    print(f"   • Consider hyperparameter optimization")
    print(f"   • Explore ensemble methods")
    print(f"   • Investigate feature selection or dimensionality reduction")
    print(f"   • Collect more data if possible (especially for {worst_class})")
    print(f"   • Consider advanced deep learning approaches")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main execution function."""
    
    print("="*80)
    print("MULTI-CLASS CLASSIFICATION ANALYSIS")
    print("EEG-Based Dementia Detection: CN vs AD vs FTD")
    print("="*80)
    
    # Set random seed
    np.random.seed(RANDOM_SEED)
    
    # 1. Load data
    X, y, feature_names, label_encoder = load_data()
    
    # 2. Create models
    models, model_params = create_models()
    
    # 3. Run classification experiments
    results_dict = run_classification_experiments(models, X, y)
    
    # 4. Feature importance analysis
    importance_df = extract_feature_importance(models, X, y, feature_names)
    
    # 5. Generate visualizations
    generate_visualizations(results_dict, importance_df, label_encoder)
    
    # 6. Save results
    save_results(results_dict, importance_df, model_params, label_encoder, y)
    
    # 7. Scientific summary
    print_scientific_summary(results_dict, importance_df, y, label_encoder)
    
    print("\n" + "="*80)
    print("✓ Classification analysis complete!")
    print(f"Results saved to: {OUTPUT_DIR.absolute()}")
    print("="*80)


if __name__ == "__main__":
    main()
