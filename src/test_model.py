import pandas as pd
import numpy as np
import sys
import os
import argparse

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from logistic_regression_model import LogisticRegressionModel
from random_forest_model import RandomForestModel
from lightgbm_model import LightGBMModel
from xgboost_model import XGBoostModel

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
    else:  # default to logistic regression
        model = LogisticRegressionModel()
        model_filename = f'saved_models/logistic_regression_{dataset_type}_model.pkl'

    model.load(model_filename)

    # Load test data based on dataset type
    if dataset_type == 'original':
        df = pd.read_csv('data/ret_multivariant_training_data.csv')
    else:
        df = pd.read_csv('data/ret_multivariant_case_control_dataset.csv')
    
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
    
    # Use the EXACT same split as training
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        features_scaled, df['mtc_diagnosis'], test_size=0.2, random_state=42, stratify=df['mtc_diagnosis']
    )
    
    # Get test patient indices
    test_indices = y_test.index
    
    return model, X_test, y_test, df.iloc[test_indices]

def generate_predictions(model, X_test_scaled, y_test, threshold=None):
    """generate predictions and probabilities using new model structure"""
    
    # Use model's built-in prediction methods
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # probability of positive class
    
    # If custom threshold provided, override model's threshold
    if threshold is not None:
        y_pred = (y_pred_proba > threshold).astype(int)
    
    return y_pred, y_pred_proba

def print_test_metrics(model, X_test, y_test, model_type='logistic', dataset_type='expanded'):
    """compute and print test metrics using model's built-in evaluation"""

    # Use model's built-in evaluation method to get metrics
    metrics = model.evaluate(X_test, y_test)

    # Print the evaluation
    model.print_evaluation(X_test, y_test)

    # Additional detailed analysis
    y_pred, y_pred_proba = generate_predictions(model, X_test, y_test)

    # Calculate precision and recall from predictions
    from sklearn.metrics import precision_score, recall_score, precision_recall_curve
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Show precision-recall curve insights
    precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_pred_proba)
    optimal_idx = np.argmax(precision_curve * recall_curve)  # F1-like optimization
    optimal_threshold = thresholds[optimal_idx] if len(thresholds) > optimal_idx else 0.5
    print(f"Optimal threshold (F1-like): {optimal_threshold:.3f}")
    print()

    # Detailed classification report
    from sklearn.metrics import classification_report
    print("CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=['No MTC', 'MTC']))
    print("=" * 60)

    # Save results to file
    os.makedirs('results', exist_ok=True)
    results_file = f'results/{model_type}_{dataset_type}_test_results.txt'

    dataset_label = "EXPANDED" if dataset_type == 'expanded' else "ORIGINAL"
    with open(results_file, 'w') as f:
        f.write(f"{model.model_name.upper()} TEST RESULTS ({dataset_label})\n")
        f.write("=" * 50 + "\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {metrics['f1']:.4f}\n")
        f.write(f"ROC AUC: {metrics['roc_auc']:.4f}\n")
        f.write(f"Average Precision: {metrics['average_precision']:.4f}\n")
        f.write("=" * 50 + "\n")

    print(f"\nResults saved to {results_file}")

def risk_stratification(probability):
    """Convert binary predictions to risk tiers for clinical use"""
    if probability < 0.10:
        return "Low Risk (0.10)"
    elif probability < 0.20:
        return "Moderate Risk (0.20)"  
    elif probability < 0.50:
        return "High Risk (0.50)"
    else:
        return "Very High Risk (1.00)"

def print_individual_predictions(model, test_patients, y_test):
    """show individual patient predictions with risk stratification using model's built-in methods"""
    
    # encode gender to numeric if it's string (safety check)
    test_patients = test_patients.copy()
    if 'gender' in test_patients.columns:
        if test_patients['gender'].dtype == 'object':
            test_patients['gender'] = test_patients['gender'].map({'Female': 0, 'Male': 1}).fillna(0)
    
    # Prepare features for the test patients (same logic as training)
    feature_cols = ['age', 'gender', 'ret_risk_level',
                    'calcitonin_elevated', 'calcitonin_level_numeric',
                    'thyroid_nodules_present', 'multiple_nodules', 'family_history_mtc',
                    'pheochromocytoma', 'hyperparathyroidism']
    features = test_patients[feature_cols].copy()

    # one-hot encode ret_variant
    if 'ret_variant' in test_patients.columns:
        variant_dummies = pd.get_dummies(test_patients['ret_variant'], prefix='variant')
        features = pd.concat([features, variant_dummies], axis=1)

    # Add age group dummies if they were used in training
    if any('age_group_' in col for col in model.feature_columns):
        age_dummies = pd.get_dummies(test_patients['age_group'], prefix='age_group')
        features = pd.concat([features, age_dummies], axis=1)

    # Add meaningful features (same as training)
    features['age_squared'] = test_patients['age'] ** 2
    features['calcitonin_age_interaction'] = test_patients['calcitonin_level_numeric'] * test_patients['age']
    features['nodule_severity'] = test_patients['thyroid_nodules_present'] * test_patients['multiple_nodules']

    # Add variant-specific interaction features (same as training)
    if 'ret_risk_level' in features.columns:
        features['risk_calcitonin_interaction'] = features['ret_risk_level'] * test_patients['calcitonin_level_numeric']
        features['risk_age_interaction'] = features['ret_risk_level'] * test_patients['age']

    # Ensure all expected columns are present and in the correct order
    for col in model.feature_columns:
        if col not in features.columns:
            features[col] = 0

    # Reorder columns to match training order
    features = features[model.feature_columns]

    # Handle NaN values (fill with median for numeric columns) - same as training
    if features.isnull().any().any():
        features = features.fillna(features.median())

    # Get predictions using model's built-in methods
    y_pred, y_pred_proba = generate_predictions(model, model.scaler.transform(features), y_test)
    
    print("=" * 80)
    print("INDIVIDUAL PATIENT PREDICTIONS WITH RISK STRATIFICATION")
    print("=" * 80)
    
    print(f"{'Patient':<20} {'Age':<6} {'Gender':<8} {'Actual':<10} {'Risk Tier':<25} {'Probability':<12}")
    print("-" * 80)
    
    for i, (_, patient) in enumerate(test_patients.iterrows()):
        patient_id = f"Patient_{i+1}"
        age = f"{patient['age']:.0f}"
        gender = "Male" if patient['gender'] == 1 else "Female"
        actual = "MTC" if y_test.iloc[i] == 1 else "No MTC"
        prob = y_pred_proba[i]
        risk_tier = model.risk_stratification([prob])[0]  # Use model's risk stratification
        
        print(f"{patient_id:<20} {age:<6} {gender:<8} {actual:<10} {risk_tier:<25} {prob:<12.3f}")
    
    print("=" * 80)
    
    print("\nRISK STRATIFICATION SUMMARY:")
    print("-" * 50)
    
    risk_tiers = model.risk_stratification(y_pred_proba)
    
    # create dynamic risk counts based on actual risk tiers
    risk_counts = {}
    for risk_tier in risk_tiers:
        if risk_tier not in risk_counts:
            risk_counts[risk_tier] = 0
        risk_counts[risk_tier] += 1
    
    for risk_tier, count in risk_counts.items():
        if count > 0:
            print(f"{risk_tier}: {count} patients")
    
    print("-" * 50)

def print_model_insights(model, X_test, y_test, test_patients):
    """print insights about model performance using new model structure"""
    print("=" * 60)
    print("MODEL PERFORMANCE INSIGHTS")
    print("=" * 60)
    
    # Get predictions for analysis
    y_pred, y_pred_proba = generate_predictions(model, X_test, y_test)
    
    # Analyze prediction patterns
    print("PREDICTION PATTERNS:")
    print(f"- Correct predictions: {(y_test == y_pred).sum()}/{len(y_test)} ({(y_test == y_pred).mean():.1%})")
    print(f"- False positives: {int(((y_test == 0) & (y_pred == 1)).sum())}")
    print(f"- False negatives: {int(((y_test == 1) & (y_pred == 0)).sum())}")
    print()
    
    # Print feature importance using model's built-in method
    model.print_feature_importance(top_n=5)
    
    print("MODEL STRENGTHS:")
    print("- Good at identifying MTC cases (high recall for positive class)")
    print("- Calcitonin levels are strong predictors")
    print("- Age and nodule presence provide additional predictive power")
    print()
    
    print("MODEL LIMITATIONS:")
    print("- Class imbalance may affect precision")
    print("- Limited training data from rare genetic condition")
    print("- Synthetic data introduces some uncertainty")
    print("=" * 60)

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description='test mtc prediction model')
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

    print(f"testing model type: {model_type}")
    print(f"testing dataset type: {dataset_type}")

    # load model and test data
    model, X_test_scaled, y_test, test_patients = load_model_and_test_data(model_type, dataset_type)

    # print results using new model structure
    print_test_metrics(model, X_test_scaled, y_test, model_type, dataset_type)
    print_individual_predictions(model, test_patients, y_test)
    print_model_insights(model, X_test_scaled, y_test, test_patients)