"""
Random Forest Model for Iris Classification
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import mlflow
import mlflow.sklearn
from utils import load_data, evaluate_model, plot_confusion_matrix, log_model_to_mlflow


class RandomForestModel:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        self.is_fitted = False
    
    def train(self, X_train, y_train):
        """Train the random forest model"""
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        print("Random Forest model trained successfully!")
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        """Get feature importance"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting feature importance")
        
        return self.model.feature_importances_
    
    def get_params(self):
        """Get model parameters"""
        return {
            'model_type': 'Random Forest',
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'random_state': self.random_state,
            'criterion': self.model.criterion,
            'min_samples_split': self.model.min_samples_split,
            'min_samples_leaf': self.model.min_samples_leaf
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'params': self.get_params()
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        
        # Create new instance
        instance = cls()
        instance.model = model_data['model']
        instance.is_fitted = True
        
        return instance


def main():
    """Main training function"""
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Initialize and train model
    rf_model = RandomForestModel(n_estimators=100, max_depth=10)
    rf_model.train(X_train, y_train)
    
    # Evaluate model
    metrics, y_pred = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    # Plot confusion matrix
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    cm_fig = plot_confusion_matrix(y_test, y_pred, "Random Forest", class_names)
    
    # Feature importance
    feature_names = ['sepal length', 'sepal width', 'petal length', 'petal width']
    importance = rf_model.get_feature_importance()
    
    print("\\nFeature Importance:")
    for i, imp in enumerate(importance):
        print(f"{feature_names[i]}: {imp:.4f}")
    
    # Save model
    model_path = '../models/random_forest_model.pkl'
    rf_model.save_model(model_path)
    
    # Log to MLflow
    params = rf_model.get_params()
    # Add feature importance to params
    for i, imp in enumerate(importance):
        params[f'feature_importance_{feature_names[i].replace(" ", "_")}'] = imp
    
    artifacts = ['../results/random_forest_confusion_matrix.png']
    
    run_id = log_model_to_mlflow(
        model=rf_model.model,
        model_name="Random Forest",
        metrics=metrics,
        params=params,
        artifacts=artifacts
    )
    
    print(f"\\nMLflow run ID: {run_id}")
    
    return rf_model, metrics


if __name__ == "__main__":
    model, metrics = main()