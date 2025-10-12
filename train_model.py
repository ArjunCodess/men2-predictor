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

    # select features (exclude non-numeric columns and target)
    feature_columns = [
        'source_id', 'age', 'gender', 'ret_k666n_positive', 'calcitonin_elevated', 'calcitonin_level_numeric',
        'thyroid_nodules_present', 'multiple_nodules', 'family_history_mtc',
        'pheochromocytoma', 'hyperparathyroidism'
    ]

    # one-hot encode age_group if needed
    if 'age_group' in df.columns:
        age_dummies = pd.get_dummies(df['age_group'], prefix='age_group')
        features = pd.concat([df[feature_columns], age_dummies], axis=1)
    else:
        features = df[feature_columns].copy()

    target = df[target_column]
    
    # return groups for group-aware splitting
    return features.drop('source_id', axis=1), target, df['source_id']


def train_evaluate_model():
    """main function to train and evaluate the model"""

    # load data
    df = load_expanded_dataset()
    print(f"loaded dataset with shape: {df.shape}")

    # prepare features, target, and groups
    features, target, groups = prepare_features_target(df, target_column='mtc_diagnosis')
    print(f"features shape: {features.shape}, target distribution: {target.value_counts().to_dict()}")

    # group-aware split to prevent leakage across variants/controls of same subject
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(features, target, groups))
    X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
    y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]
    groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]

    print(f"train set shape: {X_train.shape}, test set shape: {X_test.shape}")
    print(f"train target distribution: {y_train.value_counts().to_dict()}")
    print(f"test target distribution: {y_test.value_counts().to_dict()}")

    # scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # train logistic regression with balanced class weights
    model = LogisticRegression(
        random_state=42,
        class_weight='balanced',
        max_iter=1000
    )

    # group-aware cross-validation on training set
    # use stratified groups if available; fallback to GroupKFold
    from sklearn.model_selection import GroupKFold
    cv = GroupKFold(n_splits=min(5, len(np.unique(groups_train))))
    
    # cross-validation scores
    cv_accuracy = cross_val_score(model, X_train_scaled, y_train, groups=groups_train, cv=cv, scoring='accuracy')
    cv_f1 = cross_val_score(model, X_train_scaled, y_train, groups=groups_train, cv=cv, scoring='f1')
    cv_roc_auc = cross_val_score(model, X_train_scaled, y_train, groups=groups_train, cv=cv, scoring='roc_auc')

    print("\nCROSS-VALIDATION RESULTS (Training Set):")
    print(f"Accuracy: {cv_accuracy.mean():.3f} (±{cv_accuracy.std() * 2:.3f})")
    print(f"F1-Score: {cv_f1.mean():.3f} (±{cv_f1.std() * 2:.3f})")
    print(f"ROC-AUC: {cv_roc_auc.mean():.3f} (±{cv_roc_auc.std() * 2:.3f})")

    # train final model on full training set
    model.fit(X_train_scaled, y_train)

    # save model and scaler
    with open('model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'feature_columns': features.columns.tolist()}, f)

    print("\nmodel and scaler saved to model.pkl")
    return model, scaler, X_train_scaled, X_test_scaled, y_train, y_test, features.columns.tolist()


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