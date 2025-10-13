"""
lightgbm model for mtc prediction.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from base_model import BaseModel


class LightGBMModel(BaseModel):
    """lightgbm implementation for mtc prediction."""

    def __init__(self, threshold: float = 0.5, use_gpu: bool = False):
        """
        initialize lightgbm model.

        args:
            threshold: classification threshold (default 0.5)
            use_gpu: whether to use gpu acceleration if available
        """
        super().__init__("lightgbm", threshold)
        self.use_gpu = use_gpu

    def _create_model(self, **kwargs):
        """create lgbmclassifier model with specified parameters."""
        return LGBMClassifier(**kwargs)

    def _get_model_params(self, **kwargs):
        """get lightgbm specific parameters."""
        # default parameters optimized for small medical tabular data
        default_params = {
            'n_estimators': 200,
            'num_leaves': 31,              # primary depth control in lightgbm
            'max_depth': -1,               # let num_leaves control complexity
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.0,
            'reg_lambda': 1.0,
            'min_child_samples': 20,
            'min_split_gain': 0.0,
            'subsample_freq': 1,
            'objective': 'binary',
            'random_state': 42,
            'n_jobs': -1,
            'device_type': 'gpu' if self.use_gpu else 'cpu',
            'verbose': -1                  # silence training warnings and info
        }

        # update with any provided parameters
        default_params.update(kwargs)
        return default_params

    def cross_validate(self, X_train, y_train, cv_folds: int = 3, scoring_metrics=None):
        """
        perform cross-validation on the training data.

        args:
            x_train: training features (scaled)
            y_train: training target
            cv_folds: number of cross-validation folds
            scoring_metrics: list of scoring metrics to evaluate

        returns:
            dictionary with cross-validation results
        """
        if scoring_metrics is None:
            scoring_metrics = ['accuracy', 'f1', 'roc_auc']

        # create model with default parameters
        model_params = self._get_model_params()
        model = self._create_model(**model_params)

        # use stratified k-fold for imbalanced data
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        results = {}
        for metric in scoring_metrics:
            scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=metric)
            results[metric] = {
                'mean': scores.mean(),
                'std': scores.std(),
                'scores': scores
            }

        return results

    def print_cv_results(self, X_train, y_train, cv_folds: int = 3):
        """print cross-validation results."""
        results = self.cross_validate(X_train, y_train, cv_folds)

        print(f"\n{self.model_name.upper()} cross-validation results:")
        print("=" * 50)
        for metric, data in results.items():
            print(f"{metric.upper()}: {data['mean']:.3f} (±{data['std']:.3f})")
        print("=" * 50)

    def get_feature_importance(self):
        """
        get feature importance from lightgbm.

        returns:
            dictionary with feature names and importance scores
        """
        if not self.is_trained:
            raise ValueError("model must be trained before getting feature importance")

        if self.feature_columns is None:
            return None

        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            return dict(zip(self.feature_columns, importance))

        return None

    def risk_stratification(self, probabilities):
        """
        convert probabilities to risk tiers for clinical interpretation.

        args:
            probabilities: array of prediction probabilities

        returns:
            array of risk tier strings
        """
        risk_tiers = []
        for prob in probabilities:
            if prob < 0.20:
                risk_tiers.append("low risk (0.20)")
            elif prob < 0.40:
                risk_tiers.append("moderate risk (0.40)")
            elif prob < 0.70:
                risk_tiers.append("high risk (0.70)")
            else:
                risk_tiers.append("very high risk (1.00)")

        return risk_tiers

    def predict_proba(self, X_test):
        """
        predict class probabilities ensuring feature names consistency for lightgbm.

        args:
            x_test: test features (scaled)

        returns:
            array of shape (n_samples, 2) with probabilities for each class
        """
        if not self.is_trained:
            raise ValueError("model must be trained before making predictions")

        # wrap as dataframe with feature columns if available to avoid warnings
        if self.feature_columns is not None and isinstance(X_test, np.ndarray) and X_test.shape[1] == len(self.feature_columns):
            X_df = pd.DataFrame(X_test, columns=self.feature_columns)
            return self.model.predict_proba(X_df)
        return self.model.predict_proba(X_test)

    def predict(self, X_test):
        """
        make binary predictions using the threshold, ensuring feature names consistency.
        """
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        return (y_pred_proba > self.threshold).astype(int)
