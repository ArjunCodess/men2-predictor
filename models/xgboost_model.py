"""
xgboost model for mtc prediction.
"""

import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from base_model import BaseModel


class XGBoostModel(BaseModel):
    """xgboost implementation for mtc prediction."""

    def __init__(self, threshold: float = 0.5, use_gpu: bool = False):
        """
        initialize xgboost model.

        args:
            threshold: classification threshold (default 0.5)
            use_gpu: whether to use gpu acceleration if available
        """
        super().__init__("xgboost", threshold)
        self.use_gpu = use_gpu

    def _create_model(self, **kwargs):
        """create xgbclassifier model with specified parameters."""
        return XGBClassifier(**kwargs)

    def _get_model_params(self, **kwargs):
        """get xgboost specific parameters."""
        # default parameters optimized for small medical tabular data
        default_params = {
            'n_estimators': 100,
            'max_depth': 3,
            'learning_rate': 0.05,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.5,
            'reg_lambda': 2.0,
            'min_child_weight': 1,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1,
            'tree_method': 'gpu_hist' if self.use_gpu else 'hist',
            'scale_pos_weight': 4.2  # approximate neg/pos ratio for imbalance
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
        get feature importance from xgboost.

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
