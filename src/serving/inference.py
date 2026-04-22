import os
import json
import joblib
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    os.path.join(PROJECT_ROOT, "models", "xgb_telco_churn_model.pkl")
)

FEATURE_COLUMNS_PATH = os.getenv(
    "FEATURE_COLUMNS_PATH",
    os.path.join(PROJECT_ROOT, "artifacts", "feature_columns.json")
)

THRESHOLD = float(os.getenv("CHURN_THRESHOLD", "0.40"))


if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)
print(f"Model loaded from: {MODEL_PATH}")


if not os.path.exists(FEATURE_COLUMNS_PATH):
    raise FileNotFoundError(f"Feature columns file not found: {FEATURE_COLUMNS_PATH}")

with open(FEATURE_COLUMNS_PATH, "r") as f:
    FEATURE_COLS = json.load(f)

print(f"Loaded {len(FEATURE_COLS)} feature columns.")


BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform raw customer data into the same feature format used during training.
    """

    df = df.copy()
    df.columns = df.columns.str.strip()

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col, mapping in BINARY_MAP.items():
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .map(mapping)
                .fillna(0)
                .astype(int)
            )

    object_cols = df.select_dtypes(include=["object"]).columns.tolist()

    if object_cols:
        df = pd.get_dummies(df, columns=object_cols, drop_first=True)

    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()

    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)

    df = df.reindex(columns=FEATURE_COLS, fill_value=0)

    return df


def predict(input_dict: dict) -> dict:
    """
    Predict churn for one customer.
    """

    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)

    try:
        churn_probability = float(model.predict_proba(df_enc)[:, 1][0])
        prediction = int(churn_probability >= THRESHOLD)

    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")

    label = "Likely to churn" if prediction == 1 else "Not likely to churn"

    return {
        "prediction": label,
        "churn_probability": round(churn_probability, 4),
        "threshold": THRESHOLD,
        "model_output": prediction,
    }