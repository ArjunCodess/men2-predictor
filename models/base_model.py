"""
base model class for mtc prediction system.
provides common interface for all machine learning models.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score
import pickle
import os


class BaseModel(ABC):
    """abstract base class for all mtc prediction models."""
    
    def __init__(self, model_name: str, threshold: float = 0.5):
        """
        initialize base model.
        
        args:
            model_name: name of the model (e.g., 'logistic_regression')
            threshold: classification threshold for binary predictions
        """
        self.model_name = model_name
        self.model = None
        self.scaler = None
        self.threshold = threshold
        self.feature_columns = None
        self.is_trained = False
        
    @abstractmethod
    def _create_model(self, **kwargs):
        """create the specific model instance. must be implemented by subclasses."""
        pass
        
    @abstractmethod
    def _get_model_params(self, **kwargs):
        """get model-specific parameters. must be implemented by subclasses."""
        pass
        
    def train(self, X_train, y_train, scaler, feature_columns, **kwargs):
        """
        train the model.
        
        args:
            x_train: training features (scaled)
            y_train: training target
            scaler: fitted scaler object
            feature_columns: list of feature column names
            **kwargs: model-specific parameters
        """
        self.scaler = scaler
        self.feature_columns = feature_columns
        
        # create model with specific parameters
        model_params = self._get_model_params(**kwargs)
        self.model = self._create_model(**model_params)
        
        # train the model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        print(f"{self.model_name} model trained successfully")
        
    def predict(self, X_test):
        """
        make binary predictions using the threshold.
        
        args:
            x_test: test features (scaled)
            
        returns:
            binary predictions (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("model must be trained before making predictions")
            
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        return (y_pred_proba > self.threshold).astype(int)
        
    def predict_proba(self, X_test):
        """
        predict class probabilities.
        
        args:
            x_test: test features (scaled)
            
        returns:
            array of shape (n_samples, 2) with probabilities for each class
        """
        if not self.is_trained:
            raise ValueError("model must be trained before making predictions")
            
        return self.model.predict_proba(X_test)
        
    def evaluate(self, X_test, y_test, threshold=None):
        """
        evaluate model performance.
        
        args:
            x_test: test features (scaled)
            y_test: true test labels
            threshold: optional threshold override
            
        returns:
            dictionary with evaluation metrics
        """
        if threshold is not None:
            original_threshold = self.threshold
            self.threshold = threshold
            
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        # calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'average_precision': average_precision_score(y_test, y_pred_proba)
        }
        
        # confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        metrics['confusion_matrix'] = {
            'tn': int(cm[0, 0]),
            'fp': int(cm[0, 1]),
            'fn': int(cm[1, 0]),
            'tp': int(cm[1, 1])
        }
        
        # precision-recall curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        optimal_idx = np.argmax(precision * recall)
        optimal_threshold = thresholds[optimal_idx] if len(thresholds) > optimal_idx else 0.5
        metrics['optimal_threshold'] = optimal_threshold
        
        if threshold is not None:
            self.threshold = original_threshold
            
        return metrics
        
    def print_evaluation(self, X_test, y_test, threshold=None):
        """print detailed evaluation results."""
        metrics = self.evaluate(X_test, y_test, threshold)
        
        print(f"\n{self.model_name.upper()} EVALUATION RESULTS:")
        print("=" * 50)
        print(f"accuracy: {metrics['accuracy']:.3f}")
        print(f"f1-score: {metrics['f1']:.3f}")
        print(f"roc-auc: {metrics['roc_auc']:.3f}")
        print(f"average precision: {metrics['average_precision']:.3f}")
        print()
        
        cm = metrics['confusion_matrix']
        print("confusion matrix:")
        print(f"tn: {cm['tn']:3d} | fp: {cm['fp']:3d}")
        print(f"fn: {cm['fn']:3d} | tp: {cm['tp']:3d}")
        print()
        
        print(f"optimal threshold: {metrics['optimal_threshold']:.3f}")
        print("=" * 50)
        
    def save(self, filepath):
        """
        save the trained model and associated data.
        
        args:
            filepath: path to save the model file
        """
        if not self.is_trained:
            raise ValueError("model must be trained before saving")
            
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'threshold': self.threshold,
            'model_name': self.model_name
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
            
        print(f"{self.model_name} model saved to {filepath}")
        
    def load(self, filepath):
        """
        load a trained model and associated data.
        
        args:
            filepath: path to the saved model file
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.threshold = model_data.get('threshold', 0.5)
        self.model_name = model_data.get('model_name', 'unknown')
        self.is_trained = True
        
        print(f"{self.model_name} model loaded from {filepath}")
        
    def get_feature_importance(self):
        """
        get feature importance if available.
        
        returns:
            dictionary with feature names and importance scores, or none if not available
        """
        if not self.is_trained:
            raise ValueError("model must be trained before getting feature importance")
            
        if hasattr(self.model, 'coef_'):
            # for linear models (logistic regression, svm)
            importance = np.abs(self.model.coef_[0])
        elif hasattr(self.model, 'feature_importances_'):
            # for tree-based models (random forest, xgboost)
            importance = self.model.feature_importances_
        else:
            return None
            
        if self.feature_columns is None:
            return None
            
        return dict(zip(self.feature_columns, importance))
        
    def print_feature_importance(self, top_n=10):
        """print top n most important features."""
        importance = self.get_feature_importance()
        if importance is None:
            print(f"feature importance not available for {self.model_name}")
            return
            
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\ntop {top_n} most important features ({self.model_name}):")
        print("-" * 50)
        for i, (feature, score) in enumerate(sorted_features[:top_n], 1):
            print(f"{i:2d}. {feature}: {score:.3f}")
        print("-" * 50)
