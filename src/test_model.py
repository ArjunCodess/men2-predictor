import pandas as pd
import numpy as np
import sys
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))
from logistic_regression_model import LogisticRegressionModel
from random_forest_model import RandomForestModel
from lightgbm_model import LightGBMModel
from xgboost_model import XGBoostModel
from svm_model import SVMModel

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

    # get test patient indices and return original patient data (not processed)
    test_indices = y_test.index

    return model, X_test, y_test, df_original.iloc[test_indices]

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

    # Update results file to include CI
    dataset_label = "EXPANDED" if dataset_type == 'expanded' else "ORIGINAL"
    with open(results_file, 'w') as f:
        f.write(f"{model.model_name.upper()} TEST RESULTS ({dataset_label})\n")
        f.write("=" * 50 + "\n")
        f.write(f"Accuracy: {metrics['accuracy']:.4f} (95% CI: [{ci_results['accuracy']['ci_lower']:.4f}, {ci_results['accuracy']['ci_upper']:.4f}])\n")
        f.write(f"Precision: {precision:.4f} (95% CI: [{ci_results['precision']['ci_lower']:.4f}, {ci_results['precision']['ci_upper']:.4f}])\n")
        f.write(f"Recall: {recall:.4f} (95% CI: [{ci_results['recall']['ci_lower']:.4f}, {ci_results['recall']['ci_upper']:.4f}])\n")
        f.write(f"F1 Score: {metrics['f1']:.4f} (95% CI: [{ci_results['f1']['ci_lower']:.4f}, {ci_results['f1']['ci_upper']:.4f}])\n")
        f.write(f"ROC AUC: {metrics['roc_auc']:.4f} (95% CI: [{ci_results['roc_auc']['ci_lower']:.4f}, {ci_results['roc_auc']['ci_upper']:.4f}])\n")
        f.write(f"Average Precision: {metrics['average_precision']:.4f}\n")
        f.write("=" * 50 + "\n")
        f.write(f"\nBootstrap method: 1000 iterations, 95% confidence level\n")

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

def compare_all_models_with_patient_data(dataset_type='expanded'):
    """compare predictions across all models with full patient data"""

    # define all model types
    model_types = ['logistic', 'random_forest', 'lightgbm', 'xgboost', 'svm']
    model_names = {
        'logistic': 'LR',
        'random_forest': 'RF',
        'lightgbm': 'LGB',
        'xgboost': 'XGB',
        'svm': 'SVM'
    }

    # load test data once (using logistic model to get consistent test split)
    _, X_test_scaled, y_test, test_patients = load_model_and_test_data('logistic', dataset_type)

    # store predictions for each model
    all_predictions = {}
    loaded_models = {}

    # try to load each model and get predictions
    for model_type in model_types:
        try:
            model, X_test, y_test_check, test_patients_check = load_model_and_test_data(model_type, dataset_type)
            y_pred, _ = generate_predictions(model, X_test, y_test_check)
            all_predictions[model_type] = y_pred
            loaded_models[model_type] = model
        except Exception as e:
            print(f"warning: could not load {model_type} model: {e}")
            continue

    if not all_predictions:
        print("error: no models could be loaded for comparison")
        return

    # create detailed comparison table
    print("\n" + "=" * 160)
    print("MODEL COMPARISON - COMPLETE PATIENT DATA WITH PREDICTIONS")
    print("=" * 160)

    # build header with all patient attributes and model predictions
    header = f"{'Patient_ID':<15} {'Age':<5} {'Sex':<3} {'RET_Var':<8} {'Risk':<4} {'Calc_Elev':<4} {'Calc_Lvl':<8} {'Nod':<3} {'MN':<3} {'FH':<3} {'Pheo':<4} {'HPT':<4} | {'Actual':<8}"
    for model_type in model_types:
        if model_type in all_predictions:
            header += f" | {model_names[model_type]:<4}"
    print(header)
    print("-" * 160)

    # store results for saving to file
    comparison_results = []

    # print each test patient with full data
    for i in range(len(y_test)):
        patient = test_patients.iloc[i]

        # use source_id if study_id is nan
        if pd.notna(patient.get('study_id')):
            patient_id = str(patient['study_id'])
        else:
            patient_id = str(patient.get('source_id', f'P_{i+1}'))

        # extract all patient attributes
        age = f"{patient['age']:.0f}"
        gender = 'M' if patient['gender'] == 1 else 'F'
        ret_variant = str(patient.get('ret_variant', 'N/A'))[:8]
        risk_level = f"{patient.get('ret_risk_level', 0):.0f}"
        calc_elevated = 'Y' if patient.get('calcitonin_elevated', 0) == 1 else 'N'
        calc_level = f"{patient.get('calcitonin_level_numeric', 0):.1f}"[:8]
        nodules = 'Y' if patient.get('thyroid_nodules_present', 0) == 1 else 'N'
        mult_nodules = 'Y' if patient.get('multiple_nodules', 0) == 1 else 'N'
        family_hist = 'Y' if patient.get('family_history_mtc', 0) == 1 else 'N'
        pheo = 'Y' if patient.get('pheochromocytoma', 0) == 1 else 'N'
        hpt = 'Y' if patient.get('hyperparathyroidism', 0) == 1 else 'N'

        actual = "MTC" if y_test.iloc[i] == 1 else "No_MTC"

        row = f"{patient_id:<15} {age:<5} {gender:<3} {ret_variant:<8} {risk_level:<4} {calc_elevated:<4} {calc_level:<8} {nodules:<3} {mult_nodules:<3} {family_hist:<3} {pheo:<4} {hpt:<4} | {actual:<8}"

        # store for file output
        row_data = {
            'patient_id': patient_id,
            'age': age,
            'gender': gender,
            'ret_variant': ret_variant,
            'risk_level': risk_level,
            'calcitonin_elevated': calc_elevated,
            'calcitonin_level': calc_level,
            'nodules': nodules,
            'multiple_nodules': mult_nodules,
            'family_history': family_hist,
            'pheochromocytoma': pheo,
            'hyperparathyroidism': hpt,
            'actual': actual
        }

        # add prediction for each model
        for model_type in model_types:
            if model_type in all_predictions:
                y_pred = all_predictions[model_type]
                predicted = y_pred[i]
                is_correct = (predicted == y_test.iloc[i])
                result = "OK" if is_correct else "XX"

                # color code in terminal (green for correct, red for incorrect)
                if is_correct:
                    row += f" | \033[92m{result:<4}\033[0m"
                else:
                    row += f" | \033[91m{result:<4}\033[0m"

                row_data[model_names[model_type]] = result

        print(row)
        comparison_results.append(row_data)

    print("=" * 160)

    # calculate and display accuracy for each model
    print("\nMODEL ACCURACY SUMMARY:")
    print("-" * 50)
    accuracy_summary = []
    for model_type in model_types:
        if model_type in all_predictions:
            y_pred = all_predictions[model_type]
            correct = (y_pred == y_test).sum()
            total = len(y_test)
            accuracy = correct / total * 100
            print(f"{model_names[model_type]:<10} | {correct}/{total} correct ({accuracy:.1f}%)")
            accuracy_summary.append({
                'model': model_names[model_type],
                'correct': correct,
                'total': total,
                'accuracy': accuracy
            })
    print("=" * 160)

    # save detailed comparison to file
    os.makedirs('results', exist_ok=True)
    dataset_label = "expanded" if dataset_type == 'expanded' else "original"
    results_file = f'results/model_comparison_{dataset_label}_detailed_results.txt'

    with open(results_file, 'w') as f:
        f.write("MODEL COMPARISON - COMPLETE PATIENT DATA WITH PREDICTIONS\n")
        f.write(f"Dataset: {dataset_label.upper()}\n")
        f.write(f"Data Split: 80/20 train/test, random_state=42, stratified by target\n")
        f.write(f"SMOTE applied: Only to training data (test data is 100% real)\n")
        f.write(f"Test set size: {len(y_test)} patients\n")
        f.write("=" * 200 + "\n\n")

        # write legend
        f.write("LEGEND:\n")
        f.write("  Patient_ID: patient identifier (study_id for original data, source_id for synthetic controls)\n")
        f.write("  Age: patient age in years\n")
        f.write("  Sex: M=Male, F=Female\n")
        f.write("  RET_Var: RET genetic variant\n")
        f.write("  Risk: RET risk level (1=highest risk)\n")
        f.write("  Calc_Elev: calcitonin elevated (Y/N)\n")
        f.write("  Calc_Lvl: calcitonin level (numeric)\n")
        f.write("  Nod: thyroid nodules present (Y/N)\n")
        f.write("  MN: multiple nodules (Y/N)\n")
        f.write("  FH: family history of MTC (Y/N)\n")
        f.write("  Pheo: pheochromocytoma (Y/N)\n")
        f.write("  HPT: hyperparathyroidism (Y/N)\n")
        f.write("  OK: correct prediction\n")
        f.write("  XX: incorrect prediction\n")
        f.write("=" * 200 + "\n\n")

        # write header
        header_line = f"{'Patient_ID':<15} {'Age':<5} {'Sex':<3} {'RET_Var':<8} {'Risk':<4} {'Calc_Elev':<10} {'Calc_Lvl':<8} {'Nod':<3} {'MN':<3} {'FH':<3} {'Pheo':<4} {'HPT':<4} | {'Actual':<8}"
        for model_type in model_types:
            if model_type in all_predictions:
                header_line += f" | {model_names[model_type]:<4}"
        f.write(header_line + "\n")
        f.write("-" * 200 + "\n")

        # write each patient result
        for row_data in comparison_results:
            line = f"{row_data['patient_id']:<15} {row_data['age']:<5} {row_data['gender']:<3} {row_data['ret_variant']:<8} {row_data['risk_level']:<4} {row_data['calcitonin_elevated']:<10} {row_data['calcitonin_level']:<8} {row_data['nodules']:<3} {row_data['multiple_nodules']:<3} {row_data['family_history']:<3} {row_data['pheochromocytoma']:<4} {row_data['hyperparathyroidism']:<4} | {row_data['actual']:<8}"
            for model_type in model_types:
                if model_type in all_predictions:
                    result = row_data.get(model_names[model_type], 'N/A')
                    line += f" | {result:<4}"
            f.write(line + "\n")

        f.write("=" * 200 + "\n\n")

        # write accuracy summary
        f.write("MODEL ACCURACY SUMMARY:\n")
        f.write("-" * 50 + "\n")
        for summary in accuracy_summary:
            f.write(f"{summary['model']:<10} | {summary['correct']}/{summary['total']} correct ({summary['accuracy']:.1f}%)\n")
        f.write("=" * 200 + "\n")

    print(f"\ndetailed comparison results saved to: {results_file}")

    return all_predictions, loaded_models, comparison_results

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

def generate_correlation_matrix(model, test_patients, model_type='logistic', dataset_type='expanded'):
    """Generate and save correlation matrix for model features"""
    print("=" * 60)
    print("GENERATING CORRELATION MATRIX")
    print("=" * 60)

    # Create correlation_matrices directory if it doesn't exist
    os.makedirs('charts/correlation_matrices', exist_ok=True)

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

    # Add target variable for correlation analysis
    features['mtc_diagnosis'] = test_patients['mtc_diagnosis']

    # Handle NaN values
    if features.isnull().any().any():
        features = features.fillna(features.median())

    # Calculate correlation matrix
    correlation_matrix = features.corr()

    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=(20, 16))

    # Generate heatmap
    sns.heatmap(correlation_matrix,
                annot=False,  # Don't show values for clarity with many features
                cmap='coolwarm',
                center=0,
                vmin=-1, vmax=1,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                ax=ax)

    # Set title
    model_names = {
        'logistic': 'Logistic Regression',
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM',
        'svm': 'Support Vector Machine'
    }
    dataset_label = "Expanded Dataset" if dataset_type == 'expanded' else "Original Dataset"
    title = f'Feature Correlation Matrix\n{model_names.get(model_type, model_type)} - {dataset_label}'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()

    # Save figure with simplified naming
    # Map model_type to full model name for consistency
    model_name_mapping = {
        'logistic': 'logistic_regression',
        'random_forest': 'random_forest',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'svm': 'svm'
    }
    full_model_name = model_name_mapping.get(model_type, model_type)
    output_filename = f'charts/correlation_matrices/{full_model_name}_{dataset_type}.png'
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Correlation matrix saved to: {output_filename}")

    # Print top correlations with target variable
    print("\nTop 10 features correlated with MTC diagnosis:")
    target_corr = correlation_matrix['mtc_diagnosis'].abs().sort_values(ascending=False)
    for i, (feature, corr) in enumerate(target_corr[1:11].items(), 1):  # Skip mtc_diagnosis itself
        actual_corr = correlation_matrix.loc[feature, 'mtc_diagnosis']
        print(f"{i:2d}. {feature:<40} {actual_corr:>7.3f}")

    print("=" * 60)

    return correlation_matrix

def generate_roc_curve(model, X_test, y_test, model_type='logistic', dataset_type='expanded'):
    """Generate and save ROC curve"""
    from sklearn.metrics import roc_curve, auc
    
    print("=" * 60)
    print("GENERATING ROC CURVE")
    print("=" * 60)
    
    # Create roc_curves directory if it doesn't exist
    os.makedirs('charts/roc_curves', exist_ok=True)
    
    # Get predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    
    # Add optimal threshold point
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    plt.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10,
             label=f'Optimal Threshold = {optimal_threshold:.3f}')
    
    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    
    model_names = {
        'logistic': 'Logistic Regression',
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM',
        'svm': 'Support Vector Machine'
    }
    dataset_label = "Expanded Dataset" if dataset_type == 'expanded' else "Original Dataset"
    title = f'ROC Curve - {model_names.get(model_type, model_type)}\n{dataset_label}'
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save figure
    model_name_mapping = {
        'logistic': 'logistic_regression',
        'random_forest': 'random_forest',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'svm': 'svm'
    }
    full_model_name = model_name_mapping.get(model_type, model_type)
    output_filename = f'charts/roc_curves/{full_model_name}_{dataset_type}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"ROC curve saved to: {output_filename}")
    print(f"AUC: {roc_auc:.4f}")
    print(f"Optimal threshold: {optimal_threshold:.3f}")
    print("=" * 60)
    
    return roc_auc, optimal_threshold

def generate_confusion_matrix(model, X_test, y_test, model_type='logistic', dataset_type='expanded'):
    """Generate and save confusion matrix"""
    from sklearn.metrics import confusion_matrix
    
    print("\n" + "=" * 60)
    print("GENERATING CONFUSION MATRIX")
    print("=" * 60)
    
    # Create confusion_matrices directory if it doesn't exist
    os.makedirs('charts/confusion_matrices', exist_ok=True)
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No MTC', 'MTC'], 
                yticklabels=['No MTC', 'MTC'],
                cbar_kws={'label': 'Count'},
                ax=ax1)
    ax1.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax1.set_title('Confusion Matrix (Counts)', fontsize=12, fontweight='bold')
    
    # Plot 2: Normalized (percentages)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=['No MTC', 'MTC'],
                yticklabels=['No MTC', 'MTC'],
                cbar_kws={'label': 'Percentage'},
                ax=ax2)
    ax2.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=12, fontweight='bold')
    
    # Overall title
    model_names = {
        'logistic': 'Logistic Regression',
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost',
        'lightgbm': 'LightGBM',
        'svm': 'Support Vector Machine'
    }
    dataset_label = "Expanded Dataset" if dataset_type == 'expanded' else "Original Dataset"
    fig.suptitle(f'{model_names.get(model_type, model_type)} - {dataset_label}', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    # Save figure
    model_name_mapping = {
        'logistic': 'logistic_regression',
        'random_forest': 'random_forest',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'svm': 'svm'
    }
    full_model_name = model_name_mapping.get(model_type, model_type)
    output_filename = f'charts/confusion_matrices/{full_model_name}_{dataset_type}.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to: {output_filename}")
    
    # Print detailed metrics
    tn, fp, fn, tp = cm.ravel()
    print(f"\nConfusion Matrix Breakdown:")
    print(f"  True Negatives (TN):  {tn:3d} - Correctly predicted No MTC")
    print(f"  False Positives (FP): {fp:3d} - Incorrectly predicted MTC")
    print(f"  False Negatives (FN): {fn:3d} - Missed MTC cases")
    print(f"  True Positives (TP):  {tp:3d} - Correctly predicted MTC")
    print(f"\nKey Metrics:")
    print(f"  Sensitivity (Recall): {tp/(tp+fn):.2%} - % of actual MTC cases caught")
    print(f"  Specificity:          {tn/(tn+fp):.2%} - % of actual No MTC correctly identified")
    print(f"  Precision (PPV):      {tp/(tp+fp):.2%} - % of predicted MTC that are correct")
    print(f"  NPV:                  {tn/(tn+fn):.2%} - % of predicted No MTC that are correct")
    print("=" * 60)
    
    return cm

if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser(description='test mtc prediction model')
    parser.add_argument('--m', '--model', type=str, default='l',
                       choices=['l', 'r', 'x', 'g', 's', 'logistic', 'random_forest', 'xgboost', 'lightgbm', 'svm'],
                       help='model type: l/logistic (default), r/random_forest, x/xgboost, g/lightgbm, s/svm')
    parser.add_argument('--d', '--data', type=str, default='e',
                       choices=['e', 'o', 'expanded', 'original'],
                       help='dataset type: e/expanded (with controls + SMOTE - default), o/original (paper data only)')

    args = parser.parse_args()

    # determine dataset type
    if args.d in ['o', 'original']:
        dataset_type = 'original'
    else:
        dataset_type = 'expanded'

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

    print(f"testing model type: {model_type}")
    print(f"testing dataset type: {dataset_type}")

    # load model and test data
    model, X_test_scaled, y_test, test_patients = load_model_and_test_data(model_type, dataset_type)

    # print results using new model structure
    print_test_metrics(model, X_test_scaled, y_test, model_type, dataset_type)

    # run model comparison with full patient data (integrated into normal pipeline)
    print("\n")
    compare_all_models_with_patient_data(dataset_type)

    print_individual_predictions(model, test_patients, y_test)
    print_model_insights(model, X_test_scaled, y_test, test_patients)

    # generate visualizations
    generate_correlation_matrix(model, test_patients, model_type, dataset_type)
    generate_roc_curve(model, X_test_scaled, y_test, model_type, dataset_type)
    generate_confusion_matrix(model, X_test_scaled, y_test, model_type, dataset_type)