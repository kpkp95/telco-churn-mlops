import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.data.load_data import load_data
from src.utils.validate import validate_telco_data
from src.data.pre_process import preprocess_data
from src.features.build_features import build_features


DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "telecoData.csv")
TARGET_COL = "Churn"


def main():
    print("=== Testing Phase 1: Load → Validate → Preprocess → Build Features ===")

    # 1. Load raw data
    print("\n[1] Loading raw data...")
    df = load_data(DATA_PATH)
    print(f"Raw data shape: {df.shape}")

    assert df is not None, "Dataframe should not be None"
    assert df.shape[0] > 0, "Raw dataframe has no rows"
    assert TARGET_COL in df.columns, f"{TARGET_COL} column missing in raw data"

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

    assert "customerID" not in df_clean.columns, "customerID should be removed after preprocessing"
    assert TARGET_COL in df_clean.columns, f"{TARGET_COL} missing after preprocessing"
    assert set(df_clean[TARGET_COL].unique()) <= {0, 1}, "Churn should be converted to 0/1"
    assert df_clean["TotalCharges"].dtype != "object", "TotalCharges should be numeric"

    # 4. Build features
    print("\n[4] Building features...")
    df_features = build_features(df_clean, target_col=TARGET_COL)
    print(f"Data after feature engineering shape: {df_features.shape}")

    object_cols = df_features.select_dtypes(include=["object"]).columns.tolist()
    assert len(object_cols) == 0, f"Object columns still remain: {object_cols}"

    missing_count = df_features.isnull().sum().sum()
    assert missing_count == 0, f"Missing values remain: {missing_count}"

    assert TARGET_COL in df_features.columns, f"{TARGET_COL} missing after feature engineering"

    print("\nPhase 1 pipeline completed successfully!")


if __name__ == "__main__":
    main()