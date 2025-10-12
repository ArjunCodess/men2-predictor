"""
logistic regression model for mtc prediction.
refactored from the original train_model.py implementation.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    """logistic regression implementation for mtc prediction."""
    
    def __init__(self, threshold=0.15):
        """
        initialize logistic regression model.
        
        args:
            threshold: classification threshold (default 0.15 for medical screening)
        """
        super().__init__("logistic_regression", threshold)
        
    def _create_model(self, **kwargs):
        """create logisticregression model with specified parameters."""
        return LogisticRegression(**kwargs)
        
    def _get_model_params(self, **kwargs):
        """get logisticregression specific parameters."""
        # default parameters optimized for medical data
        default_params = {
            'random_state': 42,
            'class_weight': 'balanced',
            'max_iter': 1000,
            'solver': 'liblinear'  # good for small datasets
        }
        
        # Update with any provided parameters
        default_params.update(kwargs)
        return default_params
        
    def cross_validate(self, X_train, y_train, cv_folds=3, scoring_metrics=None):
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
        
    def print_cv_results(self, X_train, y_train, cv_folds=3):
        """print cross-validation results."""
        results = self.cross_validate(X_train, y_train, cv_folds)
        
        print(f"\n{self.model_name.upper()} cross-validation results:")
        print("=" * 50)
        for metric, data in results.items():
            print(f"{metric.upper()}: {data['mean']:.3f} (±{data['std']:.3f})")
        print("=" * 50)
        
    def get_coefficients(self):
        """
        get model coefficients for feature interpretation.
        
        returns:
            dictionary with feature names and coefficients
        """
        if not self.is_trained:
            raise ValueError("model must be trained before getting coefficients")
            
        if self.feature_columns is None:
            return None
            
        coefficients = self.model.coef_[0]
        return dict(zip(self.feature_columns, coefficients))
        
    def print_coefficients(self):
        """print model coefficients with interpretation."""
        coefficients = self.get_coefficients()
        if coefficients is None:
            print("coefficients not available")
            return
            
        print(f"\n{self.model_name.upper()} coefficients:")
        print("-" * 50)
        
        # sort by absolute value for importance
        sorted_coefs = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
        
        for feature, coef in sorted_coefs:
            direction = "increases" if coef > 0 else "decreases"
            print(f"{feature}: {coef:+.3f} ({direction} mtc risk)")
        print("-" * 50)
        
    def predict_with_confidence(self, X_test):
        """
        make predictions with confidence intervals (for logistic regression).
        
        args:
            x_test: test features (scaled)
            
        returns:
            dictionary with predictions, probabilities, and confidence
        """
        if not self.is_trained:
            raise ValueError("model must be trained before making predictions")
            
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > self.threshold).astype(int)
        
        # calculate confidence based on distance from threshold
        confidence = np.abs(y_pred_proba - 0.5) * 2  # scale to 0-1
        
        return {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confidence': confidence
        }
        
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
            if prob < 0.10:
                risk_tiers.append("low risk (0.10)")
            elif prob < 0.20:
                risk_tiers.append("moderate risk (0.20)")
            elif prob < 0.50:
                risk_tiers.append("high risk (0.50)")
            else:
                risk_tiers.append("very high risk (1.00)")
                
        return risk_tiers
