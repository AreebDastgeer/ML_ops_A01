"""
MLflow Configuration and Experiment Management
"""

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os


def setup_mlflow(experiment_name="iris-classification", tracking_uri=None):
    """
    Set up MLflow tracking
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    # Set experiment
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
    except mlflow.exceptions.MlflowException:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
    
    mlflow.set_experiment(experiment_name)
    return experiment_id


def get_best_model(experiment_name="iris-classification", metric="accuracy"):
    """
    Get the best model from MLflow experiments based on specified metric
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"Experiment {experiment_name} not found")
        return None
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} DESC"],
        max_results=10
    )
    
    if not runs:
        print("No runs found in the experiment")
        return None
    
    best_run = runs[0]
    print(f"Best model run ID: {best_run.info.run_id}")
    print(f"Best {metric}: {best_run.data.metrics[metric]:.4f}")
    
    return best_run


def register_model(model_name, run_id, model_path, description=None):
    """
    Register a model in MLflow Model Registry
    """
    client = MlflowClient()
    
    # Create model URI
    model_uri = f"runs:/{run_id}/{model_path}"
    
    try:
        # Register the model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        print(f"Model {model_name} registered successfully!")
        print(f"Model version: {model_version.version}")
        
        # Add description if provided
        if description:
            client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=description
            )
        
        return model_version
    
    except Exception as e:
        print(f"Error registering model: {e}")
        return None


def transition_model_stage(model_name, version, stage):
    """
    Transition a model to a specific stage
    """
    client = MlflowClient()
    
    try:
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        print(f"Model {model_name} version {version} transitioned to {stage}")
    except Exception as e:
        print(f"Error transitioning model stage: {e}")


def list_registered_models():
    """
    List all registered models
    """
    client = MlflowClient()
    models = client.list_registered_models()
    
    if not models:
        print("No registered models found")
        return
    
    print("Registered Models:")
    for model in models:
        print(f"- {model.name}")
        for version in model.latest_versions:
            print(f"  Version {version.version}: {version.current_stage}")


def compare_runs(experiment_name="iris-classification"):
    """
    Compare all runs in an experiment
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        print(f"Experiment {experiment_name} not found")
        return None
    
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    
    if not runs:
        print("No runs found in the experiment")
        return None
    
    print(f"\\nExperiment: {experiment_name}")
    print("-" * 80)
    print(f"{'Run ID':<36} {'Model':<20} {'Accuracy':<10} {'F1-Score':<10}")
    print("-" * 80)
    
    for run in runs:
        run_id = run.info.run_id[:8] + "..."
        model_name = run.data.tags.get('mlflow.runName', 'Unknown')
        accuracy = run.data.metrics.get('accuracy', 0)
        f1_score = run.data.metrics.get('f1_score', 0)
        
        print(f"{run_id:<36} {model_name:<20} {accuracy:<10.4f} {f1_score:<10.4f}")
    
    return runs


if __name__ == "__main__":
    # Setup MLflow
    experiment_id = setup_mlflow()
    print(f"MLflow experiment setup complete. Experiment ID: {experiment_id}")
    
    # You can start MLflow UI with: mlflow ui
    print("\\nTo start MLflow UI, run: mlflow ui")