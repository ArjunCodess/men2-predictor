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

warnings.filterwarnings('ignore')


def load_expanded_dataset(dataset_type='expanded'):
    """load the dataset based on type"""
    if dataset_type == 'original':
        # Load original paper dataset without controls
        return pd.read_csv('data/ret_k666n_training_data.csv')
    else:
        # Load expanded dataset with controls (default)
        return pd.read_csv('data/men2_case_control_dataset.csv')


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
    
    # select features (exclude non-numeric columns and target)
    feature_columns = [
        'age', 'gender', 'calcitonin_elevated', 'calcitonin_level_numeric',
        'thyroid_nodules_present', 'multiple_nodules', 'family_history_mtc',
        'pheochromocytoma', 'hyperparathyroidism'
    ]

    # one-hot encode age_group if needed
    if 'age_group' in df.columns:
        age_dummies = pd.get_dummies(df['age_group'], prefix='age_group')
        features = pd.concat([df[feature_columns], age_dummies], axis=1)
    else:
        features = df[feature_columns].copy()

    # ADD MEANINGFUL FEATURES
    features['age_squared'] = df['age'] ** 2
    features['calcitonin_age_interaction'] = df['calcitonin_level_numeric'] * df['age']
    features['nodule_severity'] = df['thyroid_nodules_present'] * df['multiple_nodules']

    target = df[target_column]
    
    # return groups for group-aware splitting
    return features, target, df.get('source_id', df.index)


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
    
    # Evaluate on test set
    model.print_evaluation(X_test, y_test)
    
    # Print feature importance
    model.print_feature_importance(top_n=10)
    
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
                       choices=['l', 'r', 'x', 'g', 'logistic', 'random_forest', 'xgboost', 'lightgbm'],
                       help='model type: l/logistic (default), r/random_forest, x/xgboost, g/lightgbm')
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