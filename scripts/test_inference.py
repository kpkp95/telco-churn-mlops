import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.serving.inference import predict


def main():
    print("=== Testing Inference Pipeline ===")

    high_risk_customer = {
        "gender": "Female",
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "tenure": 1,
        "MonthlyCharges": 85.0,
        "TotalCharges": 85.0,
    }

    low_risk_customer = {
        "gender": "Male",
        "Partner": "Yes",
        "Dependents": "Yes",
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Credit card (automatic)",
        "tenure": 60,
        "MonthlyCharges": 45.0,
        "TotalCharges": 2700.0,
    }

    print("\n[1] High-risk customer prediction:")
    high_result = predict(high_risk_customer)
    print(high_result)

    assert "prediction" in high_result
    assert "churn_probability" in high_result
    assert "threshold" in high_result

    print("\n[2] Low-risk customer prediction:")
    low_result = predict(low_risk_customer)
    print(low_result)

    assert "prediction" in low_result
    assert "churn_probability" in low_result
    assert "threshold" in low_result

    print("\nInference test completed successfully!")


if __name__ == "__main__":
    main()