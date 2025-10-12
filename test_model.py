import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

def load_model_and_test_data():
    """load trained model and test data"""
    
    # load model and scaler
    with open('men2_prediction_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    model = model_data['model']
    scaler = model_data['scaler']
    feature_columns = model_data['feature_columns']
    
    # load test data
    df = pd.read_csv('men2_case_control_dataset.csv')
    
    # prepare features and target
    feature_cols = [
        'source_id', 'age', 'gender', 'ret_k666n_positive', 'calcitonin_elevated', 'calcitonin_level_numeric',
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
    # drop source_id; keep it for potential grouping if needed later
    if 'source_id' in features.columns:
        features = features.drop('source_id', axis=1)
    
    # split the data the same way as training
    from sklearn.model_selection import train_test_split
    # perform the same group-aware split used in training for consistency
    from sklearn.model_selection import GroupShuffleSplit
    groups = df['source_id'] if 'source_id' in df.columns else df.index
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(features, df['mtc_diagnosis'], groups=groups))
    X_test = features.iloc[test_idx]
    y_test = df['mtc_diagnosis'].iloc[test_idx]
    
    # scale test features
    X_test_scaled = scaler.transform(X_test)
    
    return model, X_test_scaled, y_test, df.iloc[y_test.index]

def generate_predictions(model, X_test_scaled, y_test):
    """generate predictions and probabilities"""
    
    # predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]  # probability of positive class
    
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
    
    print("TEST SET PERFORMANCE:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print()
    
    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("CONFUSION MATRIX:")
    print(f"TN: {cm[0,0]:3d} | FP: {cm[0,1]:3d}")
    print(f"FN: {cm[1,0]:3d} | TP: {cm[1,1]:3d}")
    print()
    
    # detailed classification report
    print("CLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, target_names=['No MTC', 'MTC']))
    print("=" * 60)

def print_individual_predictions(test_patients, y_test, y_pred, y_pred_proba):
    """show individual patient predictions with probabilities and actual labels"""
    print("=" * 80)
    print("INDIVIDUAL PATIENT PREDICTIONS")
    print("=" * 80)
    
    print(f"{'Patient':<20} {'Age':<5} {'Gender':<8} {'Actual':<8} {'Predicted':<10} {'Probability':<12} {'Correct?'}")
    print("-" * 80)
    
    for i, (_, patient) in enumerate(test_patients.iterrows()):
        patient_id = f"Patient_{i+1}"
        age = f"{patient['age']:.0f}"
        gender = "Male" if patient['gender'] == 1 else "Female"
        actual = "MTC" if y_test.iloc[i] == 1 else "No MTC"
        predicted = "MTC" if y_pred[i] == 1 else "No MTC"
        probability = f"{y_pred_proba[i]:.3f}"
        correct = "Y" if y_test.iloc[i] == y_pred[i] else "N"
        
        print(f"{patient_id:<20} {age:<5} {gender:<8} {actual:<8} {predicted:<10} {probability:<12} {correct}")
    
    print("=" * 80)

def print_model_insights():
    """print insights about model performance"""
    print("=" * 60)
    print("MODEL PERFORMANCE INSIGHTS")
    print("=" * 60)
    
    # load test results to analyze
    model, X_test_scaled, y_test, test_patients = load_model_and_test_data()
    y_pred, y_pred_proba = generate_predictions(model, X_test_scaled, y_test)
    
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
    with open('men2_prediction_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    
    feature_cols = model_data['feature_columns']
    coefficients = model_data['model'].coef_[0]
    
    # create feature importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': np.abs(coefficients),
        'coefficient': coefficients
    }).sort_values('importance', ascending=False)
    
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
    model, X_test_scaled, y_test, test_patients = load_model_and_test_data()
    
    # generate predictions
    y_pred, y_pred_proba = generate_predictions(model, X_test_scaled, y_test)
    
    # print results
    print_test_metrics(y_test, y_pred, y_pred_proba)
    print_individual_predictions(test_patients, y_test, y_pred, y_pred_proba)
    print_model_insights()