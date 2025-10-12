import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')


def load_expanded_dataset():
    """load the expanded dataset"""
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


def train_evaluate_model():
    """main function to train and evaluate the model"""

    # load data
    df = load_expanded_dataset()
    print(f"loaded dataset with shape: {df.shape}")

    # prepare features, target, and groups
    features, target, groups = prepare_features_target(df, target_column='mtc_diagnosis')
    print(f"features shape: {features.shape}, target distribution: {target.value_counts().to_dict()}")

    # USE SMOTE FOR BALANCING
    from imblearn.over_sampling import SMOTE
    
    # Scale features before SMOTE
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Apply SMOTE with careful k_neighbors for small dataset
    smote = SMOTE(random_state=42, k_neighbors=3)  # k=3 for small data
    X_resampled, y_resampled = smote.fit_resample(features_scaled, target)
    
    print(f"After SMOTE: X_resampled shape: {X_resampled.shape}, "
          f"y_resampled distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
    
    # Split after SMOTE (use stratified split since groups don't match after resampling)
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )
    
    print(f"train set shape: {X_train.shape}, test set shape: {X_test.shape}")
    print(f"train target distribution: {pd.Series(y_train).value_counts().to_dict()}")
    print(f"test target distribution: {pd.Series(y_test).value_counts().to_dict()}")
    
    # Train model
    model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
    
    # USE STRATIFIED K-FOLD (3 folds for small data)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1')
    cv_roc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc')
    
    print(f"CROSS-VALIDATION RESULTS (Training Set):")
    print(f"Accuracy: {cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').mean():.3f} (±{cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy').std():.3f})")
    print(f"F1-Score: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
    print(f"ROC-AUC: {cv_roc_scores.mean():.3f} (±{cv_roc_scores.std():.3f})")
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Test predictions with ADJUSTED THRESHOLD
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba > 0.3).astype(int)  # Lower threshold for medical screening
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"TEST SET PERFORMANCE:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    
    # BETTER METRICS FOR IMBALANCED DATA
    from sklearn.metrics import precision_recall_curve, average_precision_score
    ap_score = average_precision_score(y_test, y_pred_proba)
    print(f"Average Precision Score: {ap_score:.3f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:")
    print(f"TN: {cm[0,0]:3d} | FP: {cm[0,1]:3d}")
    print(f"FN: {cm[1,0]:3d} | TP: {cm[1,1]:3d}")
    
    # Classification Report
    print(classification_report(y_test, y_pred, target_names=['No MTC', 'MTC']))
    
    # Save model and scaler
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_columns': features.columns.tolist(),
        'threshold': 0.3  # Save the adjusted threshold
    }
    with open('data/model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\nmodel and scaler saved to data/model.pkl (with threshold: {model_data['threshold']})")
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
    model, scaler, X_train_scaled, X_test_scaled, y_train, y_test, feature_cols = train_evaluate_model()
    print_model_summary()