import os
import sys
import argparse
import json
import joblib
import time

from xgboost import XGBClassifier

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.data.load_data import load_data
from src.utils.validate import validate_telco_data
from src.data.pre_process import preprocess_data
from src.features.build_features import build_features
from src.models.train import split_data, train_xgboost_model
from src.models.evaluate import evaluate_model
from src.models.tune import tune_model
from src.utils.experiment_tracking import setup_mlflow, log_experiment


def save_feature_artifacts(df_features, target_col: str, artifacts_dir: str):
    """
    Save feature column names and preprocessing metadata.

    This is useful later for inference/serving because new prediction data
    must have the same columns in the same order as training data.
    """

    os.makedirs(artifacts_dir, exist_ok=True)

    feature_columns = list(df_features.drop(columns=[target_col]).columns)

    feature_columns_path = os.path.join(artifacts_dir, "feature_columns.json")
    preprocessing_path = os.path.join(artifacts_dir, "preprocessing.pkl")

    with open(feature_columns_path, "w") as f:
        json.dump(feature_columns, f, indent=4)

    preprocessing_artifact = {
        "feature_columns": feature_columns,
        "target_col": target_col,
    }

    joblib.dump(preprocessing_artifact, preprocessing_path)

    print(f"Saved feature columns to: {feature_columns_path}")
    print(f"Saved preprocessing artifact to: {preprocessing_path}")

    return feature_columns


def main(args):
    print("=== Running Full Telco Churn ML Pipeline ===")

    # 1. Load raw data
    print("\n[1] Loading raw data...")
    df = load_data(args.input)
    print(f"Raw data shape: {df.shape}")

    # 2. Validate raw data
    print("\n[2] Validating raw data...")
    is_valid, errors = validate_telco_data(df)

    print("Validation result:", is_valid)
    print("Validation errors:", errors)

    if not is_valid:
        raise ValueError(f"Raw data validation failed: {errors}")

    # 3. Preprocess data
    print("\n[3] Preprocessing data...")
    df_clean = preprocess_data(df, target_col=args.target)
    print(f"Data after preprocessing shape: {df_clean.shape}")

    # 4. Build features
    print("\n[4] Building features...")
    df_features = build_features(df_clean, target_col=args.target)
    print(f"Feature dataframe shape: {df_features.shape}")

    # Final checks
    object_cols = df_features.select_dtypes(include=["object"]).columns.tolist()
    if object_cols:
        raise ValueError(f"Object columns still remain after feature engineering: {object_cols}")

    missing_count = df_features.isnull().sum().sum()
    if missing_count > 0:
        raise ValueError(f"Missing values remain after feature engineering: {missing_count}")

    # 5. Save processed dataset
    print("\n[5] Saving processed dataset...")
    processed_path = os.path.join(PROJECT_ROOT, "data", "processed", "telco_churn_processed.csv")
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_features.to_csv(processed_path, index=False)
    print(f"Processed dataset saved to: {processed_path}")

    # 6. Save feature artifacts
    print("\n[6] Saving feature artifacts...")
    artifacts_dir = os.path.join(PROJECT_ROOT, "artifacts")
    save_feature_artifacts(
        df_features=df_features,
        target_col=args.target,
        artifacts_dir=artifacts_dir,
    )

    # 7. Split data
    print("\n[7] Splitting data...")
    X_train, X_test, y_train, y_test = split_data(
        df=df_features,
        target_col=args.target,
        test_size=args.test_size,
        random_state=42,
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    # 8. Train model
    if args.tune:
        print("\n[8] Running Optuna tuning...")
        best_params = tune_model(
            X=X_train,
            y=y_train,
            scoring=args.scoring,
            n_trials=args.n_trials,
            random_state=42,
        )

        print("\nTraining tuned XGBoost model...")
        model = XGBClassifier(**best_params)

        start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_train

    else:
        print("\n[8] Training default XGBoost model...")

        start_train = time.time()
        model = train_xgboost_model(
            X_train=X_train,
            y_train=y_train,
            random_state=42,
        )
        train_time = time.time() - start_train

        best_params = model.get_params()

    print(f"Model trained in {train_time:.2f} seconds")

    # 9. Evaluate model
    print("\n[9] Evaluating model...")
    start_pred = time.time()

    metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        threshold=args.threshold,
    )

    pred_time = time.time() - start_pred

    metrics["train_time"] = train_time
    metrics["pred_time"] = pred_time

    print(f"Prediction/evaluation time: {pred_time:.4f} seconds")

    # 10. Save model locally
    print("\n[10] Saving model locally...")
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, "xgb_telco_churn_model.pkl")
    joblib.dump(model, model_path)

    print(f"Model saved to: {model_path}")

    # 11. Log MLflow experiment
    print("\n[11] Logging experiment to MLflow...")

    setup_mlflow(
        experiment_name=args.experiment,
        tracking_dir="mlruns",
    )

    params = best_params.copy()
    params["threshold"] = args.threshold
    params["test_size"] = args.test_size
    params["tuned"] = args.tune

    log_experiment(
        model=model,
        params=params,
        metrics=metrics,
        run_name=args.run_name,
    )

    print("\nFull pipeline completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Telco churn ML pipeline")

    parser.add_argument(
        "--input",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "raw", "telecoData.csv"),
        help="Path to raw CSV file",
    )

    parser.add_argument(
        "--target",
        type=str,
        default="Churn",
        help="Target column name",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Classification threshold for churn prediction",
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Test split size",
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default="Telco Churn - XGBoost",
        help="MLflow experiment name",
    )

    parser.add_argument(
        "--run_name",
        type=str,
        default="full_pipeline_xgboost",
        help="MLflow run name",
    )

    parser.add_argument(
        "--tune",
        action="store_true",
        help="Use Optuna tuning before training final model",
    )

    parser.add_argument(
        "--n_trials",
        type=int,
        default=10,
        help="Number of Optuna trials if tuning is enabled",
    )

    parser.add_argument(
        "--scoring",
        type=str,
        default="f1",
        help="Optuna scoring metric: f1 or recall",
    )

    args = parser.parse_args()
    main(args)