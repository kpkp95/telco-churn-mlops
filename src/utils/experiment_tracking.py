import os
from typing import Dict, Any

import mlflow
import mlflow.xgboost
from xgboost import XGBClassifier


def setup_mlflow(
    experiment_name: str = "Telco Churn - XGBoost",
    tracking_dir: str = "mlruns"
) -> None:
    """
    Set up MLflow tracking.

    Args:
        experiment_name: Name of the MLflow experiment.
        tracking_dir: Folder where MLflow runs will be saved.
    """

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    tracking_path = os.path.join(project_root, tracking_dir)

    mlflow.set_tracking_uri(f"file:///{tracking_path}")
    mlflow.set_experiment(experiment_name)

    print(f"MLflow tracking URI set to: {tracking_path}")
    print(f"MLflow experiment set to: {experiment_name}")


def log_experiment(
    model: XGBClassifier,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    run_name: str = "xgboost_telco_churn"
) -> None:
    """
    Log model parameters, metrics, and trained model to MLflow.

    Args:
        model: Trained XGBoost model.
        params: Model parameters to log.
        metrics: Evaluation metrics to log.
        run_name: Name of the MLflow run.
    """

    with mlflow.start_run(run_name=run_name):
        # Log model parameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        # Log evaluation metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        # Log trained model
        mlflow.xgboost.log_model(model, artifact_path="model")

        print("MLflow experiment logged successfully.")