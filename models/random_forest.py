"""
random forest model for mtc prediction.
ensemble method that often performs better than logistic regression on tabular data.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from base_model import BaseModel


class RandomForestModel(BaseModel):
    """random forest implementation for mtc prediction."""
    
    def __init__(self, threshold=0.5):
        """
        initialize random forest model.
        
        args:
            threshold: classification threshold (default 0.5)
        """
        super().__init__("random_forest", threshold)
        
    def _create_model(self, **kwargs):
        """create randomforestclassifier model with specified parameters."""
        return RandomForestClassifier(**kwargs)
        
    def _get_model_params(self, **kwargs):
        """get randomforest specific parameters."""
        # default parameters optimized for medical data
        default_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42,
            'class_weight': 'balanced',
            'n_jobs': -1  # use all available cores
        }
        
        # update with any provided parameters
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
        
    def get_feature_importance(self):
        """
        get feature importance from random forest.
        
        returns:
            dictionary with feature names and importance scores
        """
        if not self.is_trained:
            raise ValueError("model must be trained before getting feature importance")
            
        if self.feature_columns is None:
            return None
            
        importance = self.model.feature_importances_
        return dict(zip(self.feature_columns, importance))
        
    def print_feature_importance(self, top_n=10):
        """print top n most important features with detailed analysis."""
        importance = self.get_feature_importance()
        if importance is None:
            print("feature importance not available")
            return
            
        sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\ntop {top_n} most important features ({self.model_name}):")
        print("-" * 50)
        for i, (feature, score) in enumerate(sorted_features[:top_n], 1):
            print(f"{i:2d}. {feature}: {score:.3f}")
        print("-" * 50)
        
        # additional analysis
        total_importance = sum(importance.values())
        print(f"\nfeature importance analysis:")
        print(f"total importance: {total_importance:.3f}")
        print(f"top 3 features account for: {sum([score for _, score in sorted_features[:3]])/total_importance:.1%} of total importance")
        
    def get_tree_depth_stats(self):
        """
        get statistics about tree depths in the forest.
        
        returns:
            dictionary with depth statistics
        """
        if not self.is_trained:
            raise ValueError("model must be trained before getting tree statistics")
            
        depths = [tree.tree_.max_depth for tree in self.model.estimators_]
        return {
            'mean_depth': np.mean(depths),
            'std_depth': np.std(depths),
            'min_depth': np.min(depths),
            'max_depth': np.max(depths)
        }
        
    def print_tree_stats(self):
        """print random forest tree statistics."""
        stats = self.get_tree_depth_stats()
        print(f"\n{self.model_name.upper()} tree statistics:")
        print("-" * 50)
        print(f"number of trees: {self.model.n_estimators}")
        print(f"mean tree depth: {stats['mean_depth']:.1f} ± {stats['std_depth']:.1f}")
        print(f"depth range: {stats['min_depth']} - {stats['max_depth']}")
        print(f"max features per split: {self.model.max_features}")
        print("-" * 50)
        
    def predict_with_uncertainty(self, X_test):
        """
        make predictions with uncertainty estimation using tree voting.
        
        args:
            x_test: test features (scaled)
            
        returns:
            dictionary with predictions, probabilities, and uncertainty
        """
        if not self.is_trained:
            raise ValueError("model must be trained before making predictions")
            
        # get predictions from all trees
        tree_predictions = np.array([tree.predict(X_test) for tree in self.model.estimators_])
        tree_probabilities = np.array([tree.predict_proba(X_test)[:, 1] for tree in self.model.estimators_])
        
        # calculate ensemble predictions
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        # calculate uncertainty as standard deviation across trees
        uncertainty = np.std(tree_probabilities, axis=0)
        
        # calculate agreement (percentage of trees that agree with ensemble prediction)
        agreement = np.mean(tree_predictions == y_pred.reshape(-1, 1), axis=1)
        
        return {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'uncertainty': uncertainty,
            'agreement': agreement
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
            if prob < 0.20:
                risk_tiers.append("low risk (0.20)")
            elif prob < 0.40:
                risk_tiers.append("moderate risk (0.40)")
            elif prob < 0.70:
                risk_tiers.append("high risk (0.70)")
            else:
                risk_tiers.append("very high risk (1.00)")
                
        return risk_tiers
