import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import warnings
import sys
import os
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from logistic_regression_model import LogisticRegressionModel
from random_forest_model import RandomForestModel
from xgboost_model import XGBoostModel
from lightgbm_model import LightGBMModel
from svm_model import SVMModel

warnings.filterwarnings('ignore')


def load_expanded_dataset(dataset_type='expanded'):
    """load the dataset based on type"""
    if dataset_type == 'original':
        # Load original paper dataset without controls
        return pd.read_csv('data/processed/ret_multivariant_training_data.csv')
    else:
        # Load expanded dataset with controls (default)
        return pd.read_csv('data/processed/ret_multivariant_case_control_dataset.csv')


def prepare_features_target(df, target_column='mtc_diagnosis'):
    """prepare features and target for modeling"""

    # REMOVE CONSTANT FEATURES
    df = df.copy()
    constant_features = []
    for col in df.columns:
        if df[col].nunique() == 1:  # Only one unique value
            constant_features.append(col)
    
    if constant_features:
        print(f"Removing constant features: {constant_features}")
        df = df.drop(columns=constant_features)

    # encode gender to numeric if it's string
    if 'gender' in df.columns:
        if df['gender'].dtype == 'object':
            df['gender'] = df['gender'].map({'Female': 0, 'Male': 1}).fillna(0)

    # select base features (exclude non-numeric columns and target)
    feature_columns = [
        'age', 'gender', 'ret_risk_level',
        'calcitonin_elevated', 'calcitonin_level_numeric',
        'thyroid_nodules_present', 'multiple_nodules', 'family_history_mtc',
        'pheochromocytoma', 'hyperparathyroidism'
    ]

    # Start with base features
    features = df[feature_columns].copy()

    # one-hot encode ret_variant
    if 'ret_variant' in df.columns:
        variant_dummies = pd.get_dummies(df['ret_variant'], prefix='variant')
        features = pd.concat([features, variant_dummies], axis=1)

    # one-hot encode age_group if needed
    if 'age_group' in df.columns:
        age_dummies = pd.get_dummies(df['age_group'], prefix='age_group')
        features = pd.concat([features, age_dummies], axis=1)

    # ADD MEANINGFUL FEATURES
    features['age_squared'] = df['age'] ** 2
    features['calcitonin_age_interaction'] = df['calcitonin_level_numeric'] * df['age']
    features['nodule_severity'] = df['thyroid_nodules_present'] * df['multiple_nodules']

    # Add variant-specific interaction features
    if 'ret_risk_level' in features.columns:
        features['risk_calcitonin_interaction'] = features['ret_risk_level'] * df['calcitonin_level_numeric']
        features['risk_age_interaction'] = features['ret_risk_level'] * df['age']

    target = df[target_column]
    
    # return groups for group-aware splitting
    return features, target, df.get('source_id', df.index)


def calculate_bootstrap_ci(model, X, y, n_iterations=1000, confidence=0.95):
    """Calculate bootstrap confidence intervals for model performance"""
    from sklearn.utils import resample
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': []
    }
    
    print(f"\nCalculating {confidence*100:.0f}% confidence intervals (bootstrap with {n_iterations} iterations)...")
    
    for i in range(n_iterations):
        # Resample with replacement
        X_boot, y_boot = resample(X, y, random_state=i)
        
        # Get predictions
        y_pred = model.model.predict(X_boot)
        y_pred_proba = model.model.predict_proba(X_boot)[:, 1]
        
        # Calculate metrics
        scores['accuracy'].append(accuracy_score(y_boot, y_pred))
        scores['precision'].append(precision_score(y_boot, y_pred, zero_division=0))
        scores['recall'].append(recall_score(y_boot, y_pred, zero_division=0))
        scores['f1'].append(f1_score(y_boot, y_pred, zero_division=0))
        scores['roc_auc'].append(roc_auc_score(y_boot, y_pred_proba))
    
    # Calculate confidence intervals
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    ci_results = {}
    for metric, values in scores.items():
        ci_results[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'ci_lower': np.percentile(values, lower_percentile),
            'ci_upper': np.percentile(values, upper_percentile)
        }
    
    return ci_results


def train_evaluate_model(model_type='logistic', dataset_type='expanded'):
    """main function to train and evaluate the model using new model structure"""

    # load data
    df = load_expanded_dataset(dataset_type)
    dataset_label = "EXPANDED" if dataset_type == 'expanded' else "ORIGINAL"
    print(f"loaded dataset ({dataset_label}) with shape: {df.shape}")

    # prepare features, target, and groups
    features, target, groups = prepare_features_target(df, target_column='mtc_diagnosis')
    print(f"features shape: {features.shape}, target distribution: {target.value_counts().to_dict()}")

    # Handle NaN values (fill with median for numeric columns)
    if features.isnull().any().any():
        print(f"WARNING: Found NaN values in features. Filling with column medians.")
        features = features.fillna(features.median())

    # Scale features first
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # SPLIT FIRST (on real data only) - CRITICAL FIX
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, target, test_size=0.2, random_state=42, stratify=target
    )
    
    print(f"Original train: {X_train.shape}, test: {X_test.shape}")
    print(f"Train distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"Test distribution: {pd.Series(y_test).value_counts().to_dict()}")
    
    # THEN apply SMOTE only to training data
    from imblearn.over_sampling import SMOTE
    from collections import Counter

    # Determine k_neighbors based on minority class size
    minority_class_count = min(Counter(y_train).values())
    k_neighbors = min(5, minority_class_count - 1) if minority_class_count > 1 else 1

    print(f"Minority class count in training: {minority_class_count}, using k_neighbors={k_neighbors} for SMOTE")

    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE train: {X_train_balanced.shape}, "
          f"y_train_balanced distribution: {pd.Series(y_train_balanced).value_counts().to_dict()}")
    print(f"Test set: {X_test.shape} (NO SMOTE - REAL DATA ONLY)")
    
    # Use balanced training data
    X_train, y_train = X_train_balanced, y_train_balanced
    
    print(f"train set shape: {X_train.shape}, test set shape: {X_test.shape}")
    print(f"train target distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"test target distribution: {pd.Series(y_test).value_counts().to_dict()}")
    
    # create saved_models directory if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)

    # create and train model based on selection
    if model_type == 'random_forest' or model_type == 'r':
        model = RandomForestModel(threshold=0.5)
        model_filename = f'saved_models/random_forest_{dataset_type}_model.pkl'
        print(f"training random forest model on {dataset_label}...")
    elif model_type == 'xgboost' or model_type == 'x':
        model = XGBoostModel(threshold=0.5)
        model_filename = f'saved_models/xgboost_{dataset_type}_model.pkl'
        print(f"training xgboost model on {dataset_label}...")
    elif model_type == 'lightgbm' or model_type == 'g':
        model = LightGBMModel(threshold=0.5)
        model_filename = f'saved_models/lightgbm_{dataset_type}_model.pkl'
        print(f"training lightgbm model on {dataset_label}...")
    elif model_type == 'svm' or model_type == 's':
        model = SVMModel(threshold=0.15, kernel='linear')  # medical screening threshold, linear kernel
        model_filename = f'saved_models/svm_{dataset_type}_model.pkl'
        print(f"training svm model (linear kernel) on {dataset_label}...")
    else:  # default to logistic regression
        model = LogisticRegressionModel(threshold=0.15)  # medical screening threshold
        model_filename = f'saved_models/logistic_regression_{dataset_type}_model.pkl'
        print(f"training logistic regression model on {dataset_label}...")
    
    # Cross-validation using the model's built-in method
    model.print_cv_results(X_train, y_train, cv_folds=5)
    
    # Train the model
    model.train(X_train, y_train, scaler, features.columns.tolist())
    
    # Print model-specific information
    if hasattr(model, 'print_coefficients'):
        model.print_coefficients()
    if hasattr(model, 'print_tree_stats'):
        model.print_tree_stats()
    if hasattr(model, 'print_model_info'):
        model.print_model_info()
    if hasattr(model, 'print_support_vector_stats'):
        model.print_support_vector_stats()
    
    # Evaluate on test set
    model.print_evaluation(X_test, y_test)
    
    # Print feature importance
    model.print_feature_importance(top_n=10)

    # Calculate confidence intervals on test set
    print("\n" + "=" * 60)
    print("CONFIDENCE INTERVALS (Bootstrap Method)")
    print("=" * 60)
    
    ci_results = calculate_bootstrap_ci(model, X_test, y_test, n_iterations=1000, confidence=0.95)
    
    print(f"\n{'Metric':<15} {'Mean':<10} {'95% CI':<25}")
    print("-" * 60)
    for metric, stats in ci_results.items():
        ci_str = f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]"
        print(f"{metric.upper():<15} {stats['mean']:<10.4f} {ci_str:<25}")
    print("=" * 60)
    
    # Save CI results to file
    os.makedirs('results', exist_ok=True)
    ci_file = f'results/{model_type}_{dataset_type}_confidence_intervals.txt'
    with open(ci_file, 'w') as f:
        f.write(f"{model.model_name.upper()} - 95% CONFIDENCE INTERVALS ({dataset_label})\n")
        f.write("=" * 60 + "\n")
        f.write(f"Method: Bootstrap resampling (n={1000} iterations)\n")
        f.write(f"Test set size: {len(y_test)} patients\n\n")
        f.write(f"{'Metric':<15} {'Mean':<10} {'Std Dev':<10} {'95% CI':<25}\n")
        f.write("-" * 60 + "\n")
        for metric, stats in ci_results.items():
            ci_str = f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]"
            f.write(f"{metric.upper():<15} {stats['mean']:<10.4f} {stats['std']:<10.4f} {ci_str:<25}\n")
        f.write("=" * 60 + "\n")
    
    print(f"\nConfidence intervals saved to {ci_file}")
    
    # Save the model
    model.save(model_filename)
    
    print(f"\n{model.model_name} model saved to {model_filename}")
    return model, scaler, X_train, X_test, y_train, y_test, features.columns.tolist()


def print_model_summary():
    """print detailed model training summary"""
    print("=" * 60)
    print("MODEL TRAINING SUMMARY")
    print("=" * 60)

    # this function is called after training, so we need to reload data to show summary
    df = load_expanded_dataset()
    features, target, _ = prepare_features_target(df, target_column='mtc_diagnosis')

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target
    )

    print(f"Dataset size: {len(df)} patients")
    print(f"Training set: {len(X_train)} patients")
    print(f"Test set: {len(X_test)} patients")
    print(f"Features used: {len(features.columns)}")
    print(f"Target distribution: {target.value_counts().to_dict()}")
    print()

    print("MODEL CONFIGURATION:")
    print("- Algorithm: Logistic Regression")
    print("- Class weighting: Balanced")
    print("- Cross-validation: 5-fold stratified")
    print("- Random seed: 42")
    print()

    print("FEATURES INCLUDED:")
    for i, col in enumerate(features.columns, 1):
        print(f"{i:2d}. {col}")
    print("=" * 60)


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description='train mtc prediction model')
    parser.add_argument('--m', '--model', type=str, default='l',
                       choices=['l', 'r', 'x', 'g', 's', 'logistic', 'random_forest', 'xgboost', 'lightgbm', 'svm'],
                       help='model type: l/logistic (default), r/random_forest, x/xgboost, g/lightgbm, s/svm')
    parser.add_argument('--d', '--data', type=str, default='e',
                       choices=['e', 'o', 'expanded', 'original'],
                       help='dataset type: e/expanded (with controls + SMOTE - default), o/original (paper data only)')

    args = parser.parse_args()

    # determine model type
    if args.m in ['r', 'random_forest']:
        model_type = 'random_forest'
    elif args.m in ['x', 'xgboost']:
        model_type = 'xgboost'
    elif args.m in ['g', 'lightgbm']:
        model_type = 'lightgbm'
    elif args.m in ['s', 'svm']:
        model_type = 'svm'
    else:
        model_type = 'logistic'

    # determine dataset type
    if args.d in ['o', 'original']:
        dataset_type = 'original'
    else:
        dataset_type = 'expanded'

    print(f"selected model type: {model_type}")
    print(f"selected dataset type: {dataset_type}")

    model, scaler, X_train_scaled, X_test_scaled, y_train, y_test, feature_cols = train_evaluate_model(model_type, dataset_type)
    print_model_summary()