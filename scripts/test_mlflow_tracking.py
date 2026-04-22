import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.data.load_data import load_data
from src.utils.validate import validate_telco_data
from src.data.pre_process import preprocess_data
from src.features.build_features import build_features
from src.models.train import split_data, train_xgboost_model
from src.models.evaluate import evaluate_model
from src.utils.experiment_tracking import setup_mlflow, log_experiment


DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "telecoData.csv")
TARGET_COL = "Churn"
THRESHOLD = 0.3


def main():
    print("=== Testing MLflow Experiment Tracking ===")

    # 1. Load data
    df = load_data(DATA_PATH)

    # 2. Validate
    is_valid, errors = validate_telco_data(df)
    if not is_valid:
        raise ValueError(f"Raw data validation failed: {errors}")

    # 3. Preprocess
    df_clean = preprocess_data(df, target_col=TARGET_COL)

    # 4. Build features
    df_features = build_features(df_clean, target_col=TARGET_COL)

    # 5. Split data
    X_train, X_test, y_train, y_test = split_data(
        df=df_features,
        target_col=TARGET_COL,
        test_size=0.2,
        random_state=42,
    )

    # 6. Train model
    model = train_xgboost_model(
        X_train=X_train,
        y_train=y_train,
        random_state=42,
    )

    # 7. Evaluate model
    metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        threshold=THRESHOLD,
    )

    # 8. Get model parameters
    params = model.get_params()
    params["threshold"] = THRESHOLD

    # 9. Set up MLflow
    setup_mlflow(
        experiment_name="Telco Churn - XGBoost",
        tracking_dir="mlruns",
    )

    # 10. Log experiment
    log_experiment(
        model=model,
        params=params,
        metrics=metrics,
        run_name="default_xgboost_threshold_0_3",
    )

    print("MLflow tracking test completed successfully!")


if __name__ == "__main__":
    main()