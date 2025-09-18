"""
Utility functions for MLOps Assignment 1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


def load_data():
    """Load the preprocessed train/test splits"""
    X_train = np.load('../data/X_train.npy')
    X_test = np.load('../data/X_test.npy')
    y_train = np.load('../data/y_train.npy')
    y_test = np.load('../data/y_test.npy')
    return X_train, X_test, y_train, y_test


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a trained model and return metrics
    """
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return metrics, y_pred


def plot_confusion_matrix(y_true, y_pred, model_name, class_names=None):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(f'../results/{model_name.lower().replace(" ", "_")}_confusion_matrix.png', 
                dpi=300, bbox_inches='tight')
    
    return plt.gcf()


def log_model_to_mlflow(model, model_name, metrics, params, artifacts=None):
    """
    Log model, metrics, parameters, and artifacts to MLflow
    """
    with mlflow.start_run(run_name=model_name):
        # Log parameters
        for param, value in params.items():
            mlflow.log_param(param, value)
        
        # Log metrics
        for metric, value in metrics.items():
            mlflow.log_metric(metric, value)
        
        # Log model
        mlflow.sklearn.log_model(model, model_name.lower().replace(" ", "_"))
        
        # Log artifacts
        if artifacts:
            for artifact_path in artifacts:
                mlflow.log_artifact(artifact_path)
        
        return mlflow.active_run().info.run_id


def compare_models(results_dict):
    """
    Create comparison plots for multiple models
    """
    models = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Create comparison dataframe
    comparison_data = []
    for model in models:
        for metric in metrics:
            comparison_data.append({
                'Model': model,
                'Metric': metric.replace('_', ' ').title(),
                'Value': results_dict[model][metric]
            })
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Plot comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_comparison, x='Metric', y='Value', hue='Model')
    plt.title('Model Performance Comparison')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('../results/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df_comparison