import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.data.load_data import load_data
from src.utils.validate import validate_telco_data
from src.data.pre_process import preprocess_data
from src.features.build_features import build_features


RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "telecoData.csv")
PROCESSED_DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "telco_churn_processed.csv"
)

TARGET_COL = "Churn"


def main():
    print("=== Preparing Processed Telco Churn Dataset ===")

    # 1. Load raw data
    print("\n[1] Loading raw data...")
    df = load_data(RAW_DATA_PATH)
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
    df_clean = preprocess_data(df, target_col=TARGET_COL)
    print(f"Data after preprocessing shape: {df_clean.shape}")

    # Sanity checks after preprocessing
    assert TARGET_COL in df_clean.columns, f"{TARGET_COL} column missing after preprocessing"
    assert df_clean[TARGET_COL].isna().sum() == 0, "Churn has NaNs after preprocessing"
    assert set(df_clean[TARGET_COL].unique()) <= {0, 1}, "Churn is not 0/1 after preprocessing"

    # 4. Build features
    print("\n[4] Building features...")
    df_processed = build_features(df_clean, target_col=TARGET_COL)
    print(f"Processed data shape: {df_processed.shape}")

    # Final checks before saving
    object_cols = df_processed.select_dtypes(include=["object"]).columns.tolist()
    assert len(object_cols) == 0, f"Object columns still remain: {object_cols}"

    missing_count = df_processed.isnull().sum().sum()
    assert missing_count == 0, f"Missing values remain: {missing_count}"

    # 5. Save processed dataset
    print("\n[5] Saving processed dataset...")
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df_processed.to_csv(PROCESSED_DATA_PATH, index=False)

    print(f"\nProcessed dataset saved to: {PROCESSED_DATA_PATH}")
    print(f"Final shape: {df_processed.shape}")


if __name__ == "__main__":
    main()