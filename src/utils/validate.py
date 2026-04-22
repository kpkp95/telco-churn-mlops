import pandas as pd
from typing import Tuple, List


def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate raw Telco Customer Churn data before preprocessing.

    Returns:
        Tuple[bool, List[str]]:
        - True/False depending on whether validation passed
        - list of validation errors
    """

    print("Starting raw Telco data validation...")

    errors = []

    required_columns = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    ]

    # 1. Check required columns
    for col in required_columns:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    # Stop early if important columns are missing
    if errors:
        print("Raw data validation failed.")
        return False, errors

    # 2. Check customerID
    if df["customerID"].isna().any():
        errors.append("customerID contains missing values")

    if df["customerID"].duplicated().any():
        errors.append("customerID contains duplicate values")

    # 3. Check expected categorical values
    expected_values = {
        "gender": ["Male", "Female"],
        "Partner": ["Yes", "No"],
        "Dependents": ["Yes", "No"],
        "PhoneService": ["Yes", "No"],
        "MultipleLines": ["Yes", "No", "No phone service"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["Yes", "No", "No internet service"],
        "OnlineBackup": ["Yes", "No", "No internet service"],
        "DeviceProtection": ["Yes", "No", "No internet service"],
        "TechSupport": ["Yes", "No", "No internet service"],
        "StreamingTV": ["Yes", "No", "No internet service"],
        "StreamingMovies": ["Yes", "No", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        "Churn": ["Yes", "No"],
    }

    for col, allowed_values in expected_values.items():
        invalid_values = set(df[col].dropna().unique()) - set(allowed_values)

        if invalid_values:
            errors.append(f"{col} has invalid values: {invalid_values}")

    # 4. Check numeric columns
    if df["tenure"].isna().any():
        errors.append("tenure contains missing values")

    if df["MonthlyCharges"].isna().any():
        errors.append("MonthlyCharges contains missing values")

    if (df["tenure"] < 0).any():
        errors.append("tenure contains negative values")

    if (df["tenure"] > 120).any():
        errors.append("tenure has values greater than 120 months")

    if (df["MonthlyCharges"] < 0).any():
        errors.append("MonthlyCharges contains negative values")

    if (df["MonthlyCharges"] > 200).any():
        errors.append("MonthlyCharges has unusually high values above 200")

    # 5. Check TotalCharges can become numeric
    total_charges_numeric = pd.to_numeric(df["TotalCharges"], errors="coerce")

    invalid_total_charges = total_charges_numeric.isna() & df["TotalCharges"].astype(str).str.strip().ne("")

    if invalid_total_charges.any():
        errors.append("TotalCharges contains non-numeric values that are not blank")

    if (total_charges_numeric.dropna() < 0).any():
        errors.append("TotalCharges contains negative values")

    passed = len(errors) == 0

    if passed:
        print("Raw data validation passed.")
    else:
        print("Raw data validation failed.")
        for error in errors:
            print(f"- {error}")

    return passed, errors