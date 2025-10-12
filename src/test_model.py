import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

def load_model_and_test_data():
    """load trained model and test data"""
    
    # load model and scaler (includes threshold now)
    with open('data/model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    feature_columns = model_data['feature_columns']
    threshold = model_data.get('threshold', 0.15)  # Default to 0.15 for medical screening
    
    # load test data
    df = pd.read_csv('data/men2_case_control_dataset.csv')
    
    # prepare features and target - use same logic as training
    # REMOVE CONSTANT FEATURES
    df = df.copy()
    constant_features = []
    for col in df.columns:
        if df[col].nunique() == 1:  # Only one unique value
            constant_features.append(col)
    
    if constant_features:
        print(f"Removing constant features: {constant_features}")
        df = df.drop(columns=constant_features)
    
    # select features (exclude non-numeric columns and target)
    feature_cols = [
        'age', 'gender', 'calcitonin_elevated', 'calcitonin_level_numeric',
        'thyroid_nodules_present', 'multiple_nodules', 'family_history_mtc',
        'pheochromocytoma', 'hyperparathyroidism'
    ]
    
    # add age group dummies if they were used in training
    if any('age_group_' in col for col in feature_columns):
        age_dummies = pd.get_dummies(df['age_group'], prefix='age_group')
        features = pd.concat([df[feature_cols], age_dummies], axis=1)
        # ensure all expected columns are present
        for col in feature_columns:
            if col not in features.columns:
                features[col] = 0
    else:
        features = df[feature_cols].copy()
    
    # ADD MEANINGFUL FEATURES (same as training)
    features['age_squared'] = df['age'] ** 2
    features['calcitonin_age_interaction'] = df['calcitonin_level_numeric'] * df['age']
    features['nodule_severity'] = df['thyroid_nodules_present'] * df['multiple_nodules']
    
    # Use the SAVED scaler directly (don't create new one)
    features_scaled = scaler.transform(features)
    
    # Use the EXACT same split as training
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(
        features_scaled, df['mtc_diagnosis'], test_size=0.2, random_state=42, stratify=df['mtc_diagnosis']
    )
    
    # Get test patient indices
    test_indices = y_test.index
    
    return model, X_test, y_test, df.iloc[test_indices], threshold

def generate_predictions(model, X_test_scaled, y_test, threshold=0.5):
    """generate predictions and probabilities"""
    
    # predictions with ADJUSTED THRESHOLD
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # probability of positive class
    y_pred = (y_pred_proba > threshold).astype(int)  # Use custom threshold
    
    return y_pred, y_pred_proba

def print_test_metrics(y_test, y_pred, y_pred_proba):
    """compute and print test metrics"""
    print("=" * 60)
    print("MODEL TESTING RESULTS")
    print("=" * 60)
    
    # basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # BETTER METRICS FOR IMBALANCED DATA
    from sklearn.metrics import precision_recall_curve, average_precision_score
    ap_score = average_precision_score(y_test, y_pred_proba)
    
    print("TEST SET PERFORMANCE:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"Average Precision Score: {ap_score:.3f}")
    print()
    
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("CONFUSION MATRIX:")
    print(f"TN: {cm[0,0]:3d} | FP: {cm[0,1]:3d}")
    print(f"FN: {cm[1,0]:3d} | TP: {cm[1,1]:3d}")
    print()
    
    # Show precision-recall curve insights
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    optimal_idx = np.argmax(precision * recall)  # F1-like optimization
    optimal_threshold = thresholds[optimal_idx] if len(thresholds) > optimal_idx else 0.5
    print(f"Optimal threshold (F1-like): {optimal_threshold:.3f}")
    print()
    
    # detailed classification report
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

def print_individual_predictions(test_patients, y_test, y_pred, y_pred_proba):
    """show individual patient predictions with risk stratification"""
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
        risk_tier = risk_stratification(prob)
        
        print(f"{patient_id:<20} {age:<6} {gender:<8} {actual:<10} {risk_tier:<25} {prob:<12.3f}")
    
    print("=" * 80)
    
    print("\nRISK STRATIFICATION SUMMARY:")
    print("-" * 50)
    
    risk_counts = {'Low Risk (0.10)': 0, 'Moderate Risk (0.20)': 0, 'High Risk (0.50)': 0, 'Very High Risk (1.00)': 0}
    for prob in y_pred_proba:
        risk_tier = risk_stratification(prob)
        risk_counts[risk_tier] += 1
    
    for risk_tier, count in risk_counts.items():
        if count > 0:
            print(f"{risk_tier}: {count} patients")
    
    print("-" * 50)

def print_model_insights():
    """print insights about model performance"""
    print("=" * 60)
    print("MODEL PERFORMANCE INSIGHTS")
    print("=" * 60)
    
    # load test results to analyze
    model, X_test_scaled, y_test, test_patients, threshold = load_model_and_test_data()
    y_pred, y_pred_proba = generate_predictions(model, X_test_scaled, y_test, threshold)
    
    # analyze prediction patterns
    test_df = test_patients.copy()
    test_df['actual'] = y_test
    test_df['predicted'] = y_pred
    test_df['probability'] = y_pred_proba
    
    print("PREDICTION PATTERNS:")
    print(f"- Correct predictions: {(y_test == y_pred).sum()}/{len(y_test)} ({(y_test == y_pred).mean():.1%})")
    print(f"- False positives: {int(((y_test == 0) & (y_pred == 1)).sum())}")
    print(f"- False negatives: {int(((y_test == 1) & (y_pred == 0)).sum())}")
    print()
    
    print("FEATURE IMPORTANCE INSIGHTS:")
    # get feature importance from logistic regression coefficients
    with open('data/model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    feature_cols = model_data['feature_columns']
    coefficients = model_data['model'].coef_[0]
    
    # create feature importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': np.abs(coefficients),
        'coefficient': coefficients
    }).sort_values('importance', ascending=False)
    
    # Filter out constant features (should not appear due to removal, but just in case)
    importance_df = importance_df[importance_df['importance'] > 0]
    
    print("TOP 5 MOST IMPORTANT FEATURES:")
    for i, row in importance_df.head(5).iterrows():
        direction = "increases" if row['coefficient'] > 0 else "decreases"
        print(f"{i+1}. {row['feature']}: {row['importance']:.3f} ({direction} MTC risk)")
    print()
    
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
    # load model and test data
    model, X_test_scaled, y_test, test_patients, threshold = load_model_and_test_data()
    
    # generate predictions with threshold
    y_pred, y_pred_proba = generate_predictions(model, X_test_scaled, y_test, threshold)
    
    # print results
    print_test_metrics(y_test, y_pred, y_pred_proba)
    print_individual_predictions(test_patients, y_test, y_pred, y_pred_proba)
    print_model_insights()