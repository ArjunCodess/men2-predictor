"""
support vector machine (svm) model for mtc prediction.
kernel-based learning approach with maximum-margin classification.
"""

from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from base_model import BaseModel


class SVMModel(BaseModel):
    """support vector machine implementation for mtc prediction."""

    def __init__(self, threshold=0.15, kernel='rbf', probability=True):
        """
        initialize svm model.

        args:
            threshold: classification threshold (default 0.15 for medical screening)
            kernel: kernel type ('linear' or 'rbf', default 'rbf')
            probability: enable probability estimates (default True, uses calibration)
        """
        super().__init__("support_vector_machine", threshold)
        self.kernel = kernel
        self.probability_enabled = probability
        self.calibrated_model = None

    def _create_model(self, **kwargs):
        """create svc model with specified parameters."""
        # Note: We'll wrap this in CalibratedClassifierCV for better probability estimates
        base_model = SVC(**kwargs)

        # Use probability calibration for reliable probability estimates (Platt scaling)
        if self.probability_enabled:
            # CalibratedClassifierCV will be fit during training
            return base_model
        return base_model

    def _get_model_params(self, **kwargs):
        """get svm specific parameters."""
        # default parameters optimized for medical data with small sample size
        default_params = {
            'kernel': self.kernel,
            'C': 1.0,  # regularization parameter (tune via CV)
            'gamma': 'scale',  # kernel coefficient (auto-scaled by n_features)
            'class_weight': 'balanced',  # handle class imbalance
            'random_state': 42,
            'max_iter': 5000,  # increase for convergence
            'cache_size': 500  # MB of cache for kernel computation
        }

        # For RBF kernel, gamma is important
        if self.kernel == 'rbf':
            default_params['gamma'] = kwargs.get('gamma', 'scale')

        # update with any provided parameters
        default_params.update(kwargs)

        # Remove probability parameter if present (handled separately)
        default_params.pop('probability', None)

        return default_params

    def train(self, X_train, y_train, scaler=None, feature_columns=None, **kwargs):
        """
        train the svm model with probability calibration.

        args:
            X_train: training features (scaled)
            y_train: training target
            scaler: fitted StandardScaler object
            feature_columns: list of feature column names
            **kwargs: additional parameters for model training
        """
        # Store scaler and feature columns
        self.scaler = scaler
        self.feature_columns = feature_columns

        # Get model parameters
        model_params = self._get_model_params(**kwargs)

        # Create base SVM model
        base_svm = self._create_model(**model_params)

        # Wrap in CalibratedClassifierCV for better probability estimates
        if self.probability_enabled:
            # Use 3-fold CV for calibration (stratified for imbalanced data)
            self.calibrated_model = CalibratedClassifierCV(
                base_svm,
                method='sigmoid',  # Platt scaling
                cv=3,  # 3-fold stratified CV
                n_jobs=-1
            )
            self.calibrated_model.fit(X_train, y_train)
            # Store the calibrated model as the main model
            self.model = self.calibrated_model
        else:
            # Train without calibration
            base_svm.fit(X_train, y_train)
            self.model = base_svm

        self.is_trained = True

    def predict_proba(self, X_test):
        """
        predict class probabilities.

        args:
            X_test: test features (scaled)

        returns:
            array of shape (n_samples, 2) with class probabilities
        """
        if not self.is_trained:
            raise ValueError("model must be trained before making predictions")

        # Scale features if scaler is available
        if self.scaler is not None:
            X_test = self.scaler.transform(X_test)

        return self.model.predict_proba(X_test)

    def cross_validate(self, X_train, y_train, cv_folds=3, scoring_metrics=None):
        """
        perform cross-validation on the training data.

        args:
            X_train: training features (scaled)
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
        base_model = self._create_model(**model_params)

        # Wrap in calibrated classifier for CV
        if self.probability_enabled:
            model = CalibratedClassifierCV(base_model, method='sigmoid', cv=2, n_jobs=-1)
        else:
            model = base_model

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

        print(f"\n{self.model_name.upper()} ({self.kernel.upper()} kernel) cross-validation results:")
        print("=" * 50)
        for metric, data in results.items():
            print(f"{metric.upper()}: {data['mean']:.3f} (±{data['std']:.3f})")
        print("=" * 50)

    def get_support_vectors(self):
        """
        get information about support vectors.

        returns:
            dictionary with support vector statistics
        """
        if not self.is_trained:
            raise ValueError("model must be trained before getting support vectors")

        # Access the base estimator if using calibrated model
        if hasattr(self.model, 'calibrated_classifiers_'):
            # Get first calibrated classifier (they should be similar)
            base_estimator = self.model.calibrated_classifiers_[0].estimator
        else:
            base_estimator = self.model

        if not hasattr(base_estimator, 'support_vectors_'):
            return None

        n_support = base_estimator.n_support_
        support_vectors = base_estimator.support_vectors_

        return {
            'n_support_vectors': len(support_vectors),
            'n_support_per_class': n_support,
            'support_vector_percentage': len(support_vectors) / len(base_estimator.support_) * 100 if hasattr(base_estimator, 'support_') else None
        }

    def print_support_vector_stats(self):
        """print support vector statistics."""
        sv_info = self.get_support_vectors()
        if sv_info is None:
            print("support vector information not available")
            return

        print(f"\n{self.model_name.upper()} ({self.kernel.upper()} kernel) support vector statistics:")
        print("-" * 50)
        print(f"total support vectors: {sv_info['n_support_vectors']}")
        print(f"support vectors per class: {sv_info['n_support_per_class']}")
        if sv_info['support_vector_percentage'] is not None:
            print(f"support vector percentage: {sv_info['support_vector_percentage']:.1f}%")
        print("-" * 50)

    def get_model_params_info(self):
        """
        get information about model parameters.

        returns:
            dictionary with model parameter information
        """
        if not self.is_trained:
            raise ValueError("model must be trained before getting parameters")

        # Access the base estimator if using calibrated model
        if hasattr(self.model, 'calibrated_classifiers_'):
            base_estimator = self.model.calibrated_classifiers_[0].estimator
        else:
            base_estimator = self.model

        return {
            'kernel': base_estimator.kernel,
            'C': base_estimator.C,
            'gamma': base_estimator.gamma if hasattr(base_estimator, 'gamma') else 'N/A',
            'class_weight': base_estimator.class_weight
        }

    def print_model_info(self):
        """print model parameter information."""
        params = self.get_model_params_info()
        sv_info = self.get_support_vectors()

        print(f"\n{self.model_name.upper()} model configuration:")
        print("-" * 50)
        print(f"kernel: {params['kernel']}")
        print(f"C (regularization): {params['C']}")
        print(f"gamma: {params['gamma']}")
        print(f"class_weight: {params['class_weight']}")
        print(f"probability calibration: {'enabled (Platt scaling)' if self.probability_enabled else 'disabled'}")
        if sv_info:
            print(f"support vectors: {sv_info['n_support_vectors']} ({sv_info['n_support_per_class']})")
        print("-" * 50)

    def predict_with_decision_function(self, X_test):
        """
        make predictions with decision function values.

        args:
            X_test: test features (scaled)

        returns:
            dictionary with predictions, probabilities, and decision function values
        """
        if not self.is_trained:
            raise ValueError("model must be trained before making predictions")

        y_pred_proba = self.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > self.threshold).astype(int)

        # Get decision function values from base estimator
        decision_values = None
        if hasattr(self.model, 'calibrated_classifiers_'):
            base_estimator = self.model.calibrated_classifiers_[0].estimator
            if hasattr(base_estimator, 'decision_function'):
                decision_values = base_estimator.decision_function(X_test)
        elif hasattr(self.model, 'decision_function'):
            decision_values = self.model.decision_function(X_test)

        return {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'decision_function': decision_values
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
