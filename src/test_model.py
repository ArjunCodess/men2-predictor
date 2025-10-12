import pandas as pd
import numpy as np
import sys
import os
import argparse

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from logistic_regression_model import LogisticRegressionModel
from random_forest import RandomForestModel

def load_model_and_test_data(model_type='logistic'):
    """load trained model and test data using new model structure"""
    
    # Load the trained model based on type
    if model_type == 'random_forest' or model_type == 'r':
        model = RandomForestModel()
        model_filename = 'random_forest_model.pkl'
    else:  # default to logistic regression
        model = LogisticRegressionModel()
        model_filename = 'logistic_model.pkl'
    
    model.load(model_filename)
    
    # Load test data
    df = pd.read_csv('data/men2_case_control_dataset.csv')
    
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
    
    # Select features (exclude non-numeric columns and target)
    feature_cols = [
        'age', 'gender', 'calcitonin_elevated', 'calcitonin_level_numeric',
        'thyroid_nodules_present', 'multiple_nodules', 'family_history_mtc',
        'pheochromocytoma', 'hyperparathyroidism'
    ]
    
    # Add age group dummies if they were used in training
    if any('age_group_' in col for col in model.feature_columns):
        age_dummies = pd.get_dummies(df['age_group'], prefix='age_group')
        features = pd.concat([df[feature_cols], age_dummies], axis=1)
        # Ensure all expected columns are present
        for col in model.feature_columns:
            if col not in features.columns:
                features[col] = 0
    else:
        features = df[feature_cols].copy()
    
    # ADD MEANINGFUL FEATURES (same as training)
    features['age_squared'] = df['age'] ** 2
    features['calcitonin_age_interaction'] = df['calcitonin_level_numeric'] * df['age']
    features['nodule_severity'] = df['thyroid_nodules_present'] * df['multiple_nodules']
    
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

def print_test_metrics(model, X_test, y_test):
    """compute and print test metrics using model's built-in evaluation"""
    
    # Use model's built-in evaluation method
    model.print_evaluation(X_test, y_test)
    
    # Additional detailed analysis
    y_pred, y_pred_proba = generate_predictions(model, X_test, y_test)
    
    # Show precision-recall curve insights
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    optimal_idx = np.argmax(precision * recall)  # F1-like optimization
    optimal_threshold = thresholds[optimal_idx] if len(thresholds) > optimal_idx else 0.5
    print(f"Optimal threshold (F1-like): {optimal_threshold:.3f}")
    print()
    
    # Detailed classification report
    from sklearn.metrics import classification_report
    print("CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=['No MTC', 'MTC']))
    print("=" * 60)

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
    
    # Prepare features for the test patients (same logic as training)
    features = test_patients[['age', 'gender', 'calcitonin_elevated', 'calcitonin_level_numeric',
                             'thyroid_nodules_present', 'multiple_nodules', 'family_history_mtc',
                             'pheochromocytoma', 'hyperparathyroidism']].copy()
    
    # Add age group dummies if they were used in training
    if any('age_group_' in col for col in model.feature_columns):
        age_dummies = pd.get_dummies(test_patients['age_group'], prefix='age_group')
        features = pd.concat([features, age_dummies], axis=1)
    
    # Add meaningful features (same as training)
    features['age_squared'] = test_patients['age'] ** 2
    features['calcitonin_age_interaction'] = test_patients['calcitonin_level_numeric'] * test_patients['age']
    features['nodule_severity'] = test_patients['thyroid_nodules_present'] * test_patients['multiple_nodules']
    
    # Ensure all expected columns are present and in the correct order
    for col in model.feature_columns:
        if col not in features.columns:
            features[col] = 0
    
    # Reorder columns to match training order
    features = features[model.feature_columns]
    
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
                       choices=['l', 'r', 'logistic', 'random_forest'],
                       help='model type: l/logistic for logistic regression (default), r/random_forest for random forest')
    
    args = parser.parse_args()
    
    # determine model type
    if args.m in ['r', 'random_forest']:
        model_type = 'random_forest'
    else:
        model_type = 'logistic'
    
    print(f"testing model type: {model_type}")
    
    # load model and test data
    model, X_test_scaled, y_test, test_patients = load_model_and_test_data(model_type)
    
    # print results using new model structure
    print_test_metrics(model, X_test_scaled, y_test)
    print_individual_predictions(model, test_patients, y_test)
    print_model_insights(model, X_test_scaled, y_test, test_patients)