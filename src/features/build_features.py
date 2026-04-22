import pandas as pd


def _map_binary_series(s: pd.Series) -> pd.Series:
    """
    Apply deterministic binary encoding to 2-category features.

    This converts categorical columns with exactly 2 values into 0/1 integers.
    """

    # Convert to pandas string type safely
    # This keeps missing values as missing values instead of turning them into "nan"
    s_clean = s.astype("string").str.strip()

    vals = list(s_clean.dropna().unique())
    valset = set(vals)

    # Yes/No mapping
    if valset == {"Yes", "No"}:
        return s_clean.map({"No": 0, "Yes": 1}).astype("Int64")

    # Gender mapping
    if valset == {"Male", "Female"}:
        return s_clean.map({"Female": 0, "Male": 1}).astype("Int64")

    # Generic binary mapping using alphabetical order
    if len(vals) == 2:
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return s_clean.map(mapping).astype("Int64")

    # Return unchanged if not binary
    return s


def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Build ML-ready features for the Telco churn dataset.

    Steps:
    - Identify categorical columns
    - Binary encode 2-category columns
    - One-hot encode multi-category columns
    - Convert boolean columns to integers
    - Fill remaining missing numeric values
    """

    df = df.copy()

    print(f"Starting feature engineering on {df.shape[1]} columns...")

    # Find categorical columns, excluding the target column
    obj_cols = [
        col for col in df.select_dtypes(include=["object", "string"]).columns
        if col != target_col
    ]

    print(f"Found {len(obj_cols)} categorical columns")

    # Binary columns have exactly 2 unique values
    binary_cols = [
        col for col in obj_cols
        if df[col].dropna().nunique() == 2
    ]

    # Multi-category columns have more than 2 unique values
    multi_cols = [
        col for col in obj_cols
        if df[col].dropna().nunique() > 2
    ]

    print(f"Binary columns: {binary_cols}")
    print(f"Multi-category columns: {multi_cols}")

    # Binary encoding
    for col in binary_cols:
        df[col] = _map_binary_series(df[col])
        print(f"Encoded binary column: {col}")

    # One-hot encoding
    if multi_cols:
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
        print(f"Applied one-hot encoding to: {multi_cols}")

    # Convert nullable integer columns to normal integers
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    # Convert bool columns created by get_dummies to int
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f"Converted boolean columns to int: {bool_cols}")

    # Fill remaining numeric missing values
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    print(f"Feature engineering complete: {df.shape[1]} final columns")

    return df