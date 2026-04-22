import os
import sys

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(PROJECT_ROOT)

from src.data.load_data import load_data
from src.utils.validate import validate_telco_data


def test_validate_telco_data():
    """
    Test raw Telco data validation.

    Flow:
    load raw data -> validate raw data
    """

    data_path = os.path.join(PROJECT_ROOT, "data", "raw", "telecoData.csv")

    # Load raw data
    df = load_data(data_path)

    print("Raw data shape:", df.shape)

    assert df is not None, "Dataframe should not be None"
    assert df.shape[0] > 0, "Dataframe should contain rows"

    # Validate raw data
    is_valid, failed_expectations = validate_telco_data(df)

    print("Validation result:", is_valid)
    print("Failed expectations:", failed_expectations)

    assert is_valid, f"Validation failed: {failed_expectations}"

    print("validate_data.py test passed successfully!")


if __name__ == "__main__":
    test_validate_telco_data()