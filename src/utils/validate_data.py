import pandas as pd
from typing import Tuple, List


def validate_raw_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate the raw Telco churn dataset before preprocessing.

    Checks:
    - Required columns exist
    - Important columns do not contain missing values
    - Categorical columns contain expected values
    - Numeric columns have reasonable values when possible
    """

    errors = []

    required_cols = [
        "customerID",
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "InternetService",
        "Contract",
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    ]

    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing required column: {col}")

    if errors:
        return False, errors

    expected_values = {
        "gender": ["Male", "Female"],
        "Partner": ["Yes", "No"],
        "Dependents": ["Yes", "No"],
        "PhoneService": ["Yes", "No"],
        "InternetService": ["DSL", "Fiber optic", "No"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "Churn": ["Yes", "No"],
    }

    for col, allowed_values in expected_values.items():
        invalid_values = set(df[col].dropna().unique()) - set(allowed_values)
        if invalid_values:
            errors.append(f"{col} has invalid values: {invalid_values}")

    if df["customerID"].isna().any():
        errors.append("customerID contains missing values")

    if df["tenure"].isna().any():
        errors.append("tenure contains missing values")

    if df["MonthlyCharges"].isna().any():
        errors.append("MonthlyCharges contains missing values")

    if (df["tenure"] < 0).any():
        errors.append("tenure contains negative values")

    if (df["MonthlyCharges"] < 0).any():
        errors.append("MonthlyCharges contains negative values")

    if (df["tenure"] > 120).any():
        errors.append("tenure has values greater than 120 months")

    if (df["MonthlyCharges"] > 200).any():
        errors.append("MonthlyCharges has unusually high values above 200")

    passed = len(errors) == 0

    if passed:
        print("Raw data validation passed.")
    else:
        print("Raw data validation failed.")
        for error in errors:
            print(f"- {error}")

    return passed, errors