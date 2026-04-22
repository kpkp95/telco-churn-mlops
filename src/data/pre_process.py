import pandas as pd


def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Clean the Telco churn dataset.

    Steps:
    - Copy dataframe to avoid changing original data
    - Strip column names
    - Strip whitespace from text columns
    - Drop ID columns
    - Convert target column Churn to 0/1 if needed
    - Convert TotalCharges to numeric
    - Fill missing numeric values
    - Ensure SeniorCitizen is integer
    """

    df = df.copy()

    #Clean column names
    df.columns = df.columns.str.strip()

    #to strip whitespace from string/object columns
    object_cols = df.select_dtypes(include=["object"]).columns
    for col in object_cols:
        df[col] = df[col].str.strip()

    # Drop ID columns if present
    id_cols = ["customerID", "CustomerID", "customer_id"]
    existing_id_cols = [col for col in id_cols if col in df.columns]

    if existing_id_cols:
        df = df.drop(columns=existing_id_cols)

    # Convert target column to 0/1 if it is Yes/No
    if target_col in df.columns and df[target_col].dtype == "object":
        df[target_col] = df[target_col].map({"No": 0, "Yes": 1})

    # Convert TotalCharges to numeric
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # SeniorCitizen should be integer if present
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)

    # Fill missing numeric values
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df