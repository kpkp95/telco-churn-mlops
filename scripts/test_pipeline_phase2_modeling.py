import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from xgboost import XGBClassifier

from src.data.load_data import load_data
from src.utils.validate import validate_telco_data
from src.data.pre_process import preprocess_data
from src.features.build_features import build_features
from src.models.train import split_data, train_xgboost_model
from src.models.evaluate import evaluate_model
from src.models.tune import tune_model


DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "telecoData.csv")
TARGET_COL = "Churn"
THRESHOLD = 0.3


def main():
    print("=== Testing Phase 2: Modeling Pipeline ===")

    # 1. Load raw data
    print("\n[1] Loading raw data...")
    df = load_data(DATA_PATH)
    print(f"Raw data shape: {df.shape}")

    assert df is not None, "Dataframe should not be None"
    assert df.shape[0] > 0, "Raw dataframe has no rows"

    # 2. Validate raw data
    print("\n[2] Validating raw data...")
    is_valid, errors = validate_telco_data(df)

    print("Validation result:", is_valid)
    print("Validation errors:", errors)

    assert is_valid, f"Raw data validation failed: {errors}"

    # 3. Preprocess data
    print("\n[3] Preprocessing data...")
    df_clean = preprocess_data(df, target_col=TARGET_COL)
    print(f"Data after preprocessing shape: {df_clean.shape}")

    # 4. Build features
    print("\n[4] Building features...")
    df_features = build_features(df_clean, target_col=TARGET_COL)
    print(f"Feature dataframe shape: {df_features.shape}")

    # Basic feature checks
    object_cols = df_features.select_dtypes(include=["object"]).columns.tolist()
    assert len(object_cols) == 0, f"Object columns still remain: {object_cols}"
    assert TARGET_COL in df_features.columns, f"{TARGET_COL} missing after feature engineering"

    # 5. Split data
    print("\n[5] Splitting data...")
    X_train, X_test, y_train, y_test = split_data(
        df=df_features,
        target_col=TARGET_COL,
        test_size=0.2,
        random_state=42,
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    assert X_train.shape[0] > 0, "X_train has no rows"
    assert X_test.shape[0] > 0, "X_test has no rows"
    assert X_train.shape[1] == X_test.shape[1], "Train/test feature count mismatch"

    # 6. Train default XGBoost model
    print("\n[6] Training default XGBoost model...")
    model = train_xgboost_model(
        X_train=X_train,
        y_train=y_train,
        random_state=42,
    )

    assert model is not None, "Model training failed"

    # 7. Evaluate default model
    print("\n[7] Evaluating default model...")
    metrics = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        threshold=THRESHOLD,
    )

    print("Default model metrics:")
    print(metrics)

    assert isinstance(metrics, dict), "Metrics should be returned as a dictionary"
    assert "accuracy" in metrics, "Accuracy missing from metrics"
    assert "precision" in metrics, "Precision missing from metrics"
    assert "recall" in metrics, "Recall missing from metrics"
    assert "f1" in metrics, "F1 missing from metrics"
    assert "roc_auc" in metrics, "ROC-AUC missing from metrics"

    # 8. Test Optuna tuning with small number of trials
    print("\n[8] Testing Optuna tuning with 3 trials...")
    best_params = tune_model(
        X=X_train,
        y=y_train,
        scoring="f1",
        n_trials=3,
        random_state=42,
    )

    print("Best params from test tuning:")
    print(best_params)

    assert isinstance(best_params, dict), "Best params should be a dictionary"
    assert "n_estimators" in best_params, "n_estimators missing from best params"
    assert "learning_rate" in best_params, "learning_rate missing from best params"
    assert "max_depth" in best_params, "max_depth missing from best params"

    # 9. Train tuned model
    print("\n[9] Training tuned XGBoost model...")
    tuned_model = XGBClassifier(**best_params)
    tuned_model.fit(X_train, y_train)

    assert tuned_model is not None, "Tuned model training failed"

    # 10. Evaluate tuned model
    print("\n[10] Evaluating tuned model...")
    tuned_metrics = evaluate_model(
        model=tuned_model,
        X_test=X_test,
        y_test=y_test,
        threshold=THRESHOLD,
    )

    print("Tuned model metrics:")
    print(tuned_metrics)

    assert isinstance(tuned_metrics, dict), "Tuned metrics should be returned as a dictionary"
    assert "f1" in tuned_metrics, "Tuned model F1 missing"
    assert "recall" in tuned_metrics, "Tuned model recall missing"

    print("\nPhase 2 modeling pipeline completed successfully!")


if __name__ == "__main__":
    main()