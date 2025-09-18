"""
Logistic Regression Model for Iris Classification
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import mlflow
import mlflow.sklearn
from utils import load_data, evaluate_model, plot_confusion_matrix, log_model_to_mlflow


class LogisticRegressionModel:
    def __init__(self, random_state=42, max_iter=1000):
        self.random_state = random_state
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            random_state=self.random_state,
            max_iter=self.max_iter
        )
        self.is_fitted = False
    
    def train(self, X_train, y_train):
        """Train the logistic regression model"""
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        print("Logistic Regression model trained successfully!")
        return self
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_params(self):
        """Get model parameters"""
        return {
            'model_type': 'Logistic Regression',
            'random_state': self.random_state,
            'max_iter': self.max_iter,
            'solver': self.model.solver,
            'C': self.model.C
        }
    
    def save_model(self, filepath):
        """Save the trained model"""
        if not self.is_fitted:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
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
        instance.scaler = model_data['scaler']
        instance.is_fitted = True
        
        return instance


def main():
    """Main training function"""
    # Load data
    X_train, X_test, y_train, y_test = load_data()
    
    # Initialize and train model
    lr_model = LogisticRegressionModel()
    lr_model.train(X_train, y_train)
    
    # Evaluate model
    metrics, y_pred = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    
    # Plot confusion matrix
    class_names = ['Setosa', 'Versicolor', 'Virginica']
    cm_fig = plot_confusion_matrix(y_test, y_pred, "Logistic Regression", class_names)
    
    # Save model
    model_path = '../models/logistic_regression_model.pkl'
    lr_model.save_model(model_path)
    
    # Log to MLflow
    params = lr_model.get_params()
    artifacts = ['../results/logistic_regression_confusion_matrix.png']
    
    run_id = log_model_to_mlflow(
        model=lr_model.model,
        model_name="Logistic Regression",
        metrics=metrics,
        params=params,
        artifacts=artifacts
    )
    
    print(f"\nMLflow run ID: {run_id}")
    
    return lr_model, metrics


if __name__ == "__main__":
    model, metrics = main()