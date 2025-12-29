import pandas as pd
import numpy as np
import sys
import os
import argparse
from pathlib import Path

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from logistic_regression_model import LogisticRegressionModel
from random_forest_model import RandomForestModel
from lightgbm_model import LightGBMModel
from xgboost_model import XGBoostModel
from svm_model import SVMModel

def calculate_bootstrap_ci(model, X, y, n_iterations=1000, confidence=0.95):
    """Calculate bootstrap confidence intervals for model performance metrics.

    This method uses bootstrap resampling of the test set to estimate the variability
    in performance metrics due to finite sample size. Note: This provides confidence
    intervals around the point estimates but does not account for model training variability.
    """
    from sklearn.utils import resample
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

    scores = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': [],
        'avg_precision': []
    }

    print(f"\nCalculating {confidence*100:.0f}% confidence intervals (bootstrap resampling with {n_iterations} iterations)...")

    for i in range(n_iterations):
        # Resample test set with replacement to estimate metric variability
        X_boot, y_boot = resample(X, y, random_state=i)

        # Get predictions on bootstrap sample
        y_pred = model.model.predict(X_boot)
        y_pred_proba = model.model.predict_proba(X_boot)[:, 1]

        # Calculate all performance metrics
        scores['accuracy'].append(accuracy_score(y_boot, y_pred))
        scores['precision'].append(precision_score(y_boot, y_pred, zero_division=0))
        scores['recall'].append(recall_score(y_boot, y_pred, zero_division=0))
        scores['f1'].append(f1_score(y_boot, y_pred, zero_division=0))
        scores['roc_auc'].append(roc_auc_score(y_boot, y_pred_proba))
        scores['avg_precision'].append(average_precision_score(y_boot, y_pred_proba))

    # Calculate confidence intervals using percentile method
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_results = {}
    for metric, values in scores.items():
        ci_results[metric] = {
            'mean': np.mean(values),
            'std': np.std(values, ddof=1),  # Use sample standard deviation
            'median': np.median(values),
            'ci_lower': np.percentile(values, lower_percentile),
            'ci_upper': np.percentile(values, upper_percentile)
        }

    return ci_results

def load_model_and_test_data(model_type='logistic', dataset_type='expanded'):
    """load trained model and test data using new model structure"""

    # load the trained model based on type
    if model_type == 'random_forest' or model_type == 'r':
        model = RandomForestModel()
        model_filename = f'saved_models/random_forest_{dataset_type}_model.pkl'
    elif model_type == 'lightgbm' or model_type == 'g':
        model = LightGBMModel()
        model_filename = f'saved_models/lightgbm_{dataset_type}_model.pkl'
    elif model_type == 'xgboost' or model_type == 'x':
        model = XGBoostModel()
        model_filename = f'saved_models/xgboost_{dataset_type}_model.pkl'
    elif model_type == 'svm' or model_type == 's':
        model = SVMModel()
        model_filename = f'saved_models/svm_{dataset_type}_model.pkl'
    else:  # default to logistic regression
        model = LogisticRegressionModel()
        model_filename = f'saved_models/logistic_regression_{dataset_type}_model.pkl'

    model.load(model_filename)

    # load test data based on dataset type and keep original dataframe for patient info
    if dataset_type == 'original':
        df_original = pd.read_csv('data/processed/ret_multivariant_training_data.csv')
    else:
        df_original = pd.read_csv('data/processed/ret_multivariant_case_control_dataset.csv')

    df = df_original.copy()

    # Prepare features and target - use same logic as training
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

    # Select base features (exclude non-numeric columns and target)
    feature_cols = [
        'age', 'gender', 'ret_risk_level',
        'calcitonin_elevated', 'calcitonin_level_numeric',
        'thyroid_nodules_present', 'multiple_nodules', 'family_history_mtc',
        'pheochromocytoma', 'hyperparathyroidism'
    ]

    # Start with base features
    features = df[feature_cols].copy()

    # one-hot encode ret_variant
    if 'ret_variant' in df.columns:
        variant_dummies = pd.get_dummies(df['ret_variant'], prefix='variant')
        features = pd.concat([features, variant_dummies], axis=1)

    # Add age group dummies if they were used in training
    if any('age_group_' in col for col in model.feature_columns):
        age_dummies = pd.get_dummies(df['age_group'], prefix='age_group')
        features = pd.concat([features, age_dummies], axis=1)

    # ADD MEANINGFUL FEATURES (same as training)
    features['age_squared'] = df['age'] ** 2
    features['calcitonin_age_interaction'] = df['calcitonin_level_numeric'] * df['age']
    features['nodule_severity'] = df['thyroid_nodules_present'] * df['multiple_nodules']

    # Add variant-specific interaction features (same as training)
    if 'ret_risk_level' in features.columns:
        features['risk_calcitonin_interaction'] = features['ret_risk_level'] * df['calcitonin_level_numeric']
        features['risk_age_interaction'] = features['ret_risk_level'] * df['age']

    # Ensure all expected columns are present
    for col in model.feature_columns:
        if col not in features.columns:
            features[col] = 0

    # Handle NaN values (fill with median for numeric columns) - same as training
    if features.isnull().any().any():
        print(f"WARNING: Found NaN values in features. Filling with column medians.")
        features = features.fillna(features.median())

    # Use the SAVED scaler directly (don't create new one)
    features_scaled = model.scaler.transform(features)

    # use the EXACT same split as training
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        features_scaled, df['mtc_diagnosis'], test_size=0.2, random_state=42, stratify=df['mtc_diagnosis']
    )

    return model, X_test, y_test

def main():
    """Main function to calculate confidence intervals for a trained model"""
    parser = argparse.ArgumentParser(description='Calculate confidence intervals for trained model')
    parser.add_argument('--m', '--model', type=str, default='l',
                       choices=['l', 'r', 'x', 'g', 's', 'logistic', 'random_forest', 'xgboost', 'lightgbm', 'svm'],
                       help='model type: l/logistic (default), r/random_forest, x/xgboost, g/lightgbm, s/svm')
    parser.add_argument('--d', '--data', type=str, default='e',
                       choices=['e', 'o', 'expanded', 'original'],
                       help='dataset type: e/expanded (default), o/original')
    parser.add_argument('--iterations', type=int, default=1000,
                       help='number of bootstrap iterations (default: 1000)')

    args = parser.parse_args()

    # Determine dataset type
    if args.d in ['o', 'original']:
        dataset_type = 'original'
    else:
        dataset_type = 'expanded'

    # Determine model type
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

    print("=" * 70)
    print(f"CALCULATING CONFIDENCE INTERVALS - {model_type.upper()} ({dataset_type.upper()})")
    print("=" * 70)

    try:
        # Load model and test data
        model, X_test, y_test = load_model_and_test_data(model_type, dataset_type)

        # Calculate confidence intervals
        ci_results = calculate_bootstrap_ci(model, X_test, y_test, n_iterations=args.iterations, confidence=0.95)

        # Display results
        print(f"\n{'Metric':<15} {'Mean':<10} {'Std Dev':<10} {'Median':<10} {'95% CI':<25}")
        print("-" * 70)
        for metric, stats in ci_results.items():
            ci_str = f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]"
            print(f"{metric.upper():<15} {stats['mean']:<10.4f} {stats['std']:<10.4f} {stats['median']:<10.4f} {ci_str:<25}")
        print("=" * 70)

        # Save results to file
        os.makedirs('results', exist_ok=True)
        ci_file = f'results/{model_type}_{dataset_type}_confidence_intervals.txt'

        dataset_label = "EXPANDED" if dataset_type == 'expanded' else "ORIGINAL"
        with open(ci_file, 'w') as f:
            f.write(f"{model.model_name.upper()} - 95% CONFIDENCE INTERVALS ({dataset_label})\n")
            f.write("=" * 70 + "\n")
            f.write(f"Method: Bootstrap resampling (n={args.iterations} iterations)\n")
            f.write(f"Confidence Level: 95%\n")
            f.write(f"CI Method: Percentile bootstrap\n")
            f.write(f"Test set size: {len(y_test)} patients\n\n")
            f.write(f"{'Metric':<15} {'Mean':<10} {'Std Dev':<10} {'Median':<10} {'95% CI':<25}\n")
            f.write("-" * 70 + "\n")
            for metric, stats in ci_results.items():
                ci_str = f"[{stats['ci_lower']:.4f}, {stats['ci_upper']:.4f}]"
                f.write(f"{metric.upper():<15} {stats['mean']:<10.4f} {stats['std']:<10.4f} {stats['median']:<10.4f} {ci_str:<25}\n")
            f.write("=" * 70 + "\n")
            f.write("Notes:\n")
            f.write("- Std Dev: Bootstrap standard deviation (sample)\n")
            f.write("- CI: Percentile-based confidence interval\n")
            f.write("- All metrics calculated on test set using bootstrap resampling\n")

        print(f"\nConfidence intervals saved to: {ci_file}")
        print("=" * 70)
        print("CONFIDENCE INTERVAL CALCULATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)

    except Exception as e:
        print(f"ERROR: Failed to calculate confidence intervals: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
