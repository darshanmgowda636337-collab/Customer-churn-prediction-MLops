from pathlib import Path

import pandas as pd

# Your dataset path
DATA_PATH = Path(
    r"D:\Customer-Churn-Prediction-MLOps\Customer-churn-prediction-MLops\data\customers.csv"
)

REQUIRED_COLUMNS = [
    "tenure",
    "monthly_charges",
    "support_calls",
    "contract",
    "internet_service",
    "churn",
]


def validate_data(file_path=DATA_PATH):
    file_path = Path(file_path)

    # Check whether the dataset exists
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    # Load the dataset
    df = pd.read_csv(
    file_path,
    keep_default_na=False,
    na_values=[""]
)

    # Check whether the dataset is empty
    if df.empty:
        raise ValueError("Dataset is empty")

    # Check required columns
    missing_columns = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}"
        )

    # Check missing values
    missing_values = df[REQUIRED_COLUMNS].isnull().sum()
    missing_values = missing_values[missing_values > 0]

    if not missing_values.empty:
        raise ValueError(
            "Dataset contains missing values:\n"
            f"{missing_values.to_string()}"
        )

    # Check churn values
    if not df["churn"].isin([0, 1]).all():
        raise ValueError("Churn must contain only 0 and 1")

    # Check numerical values
    if (df["tenure"] < 0).any():
        raise ValueError("Tenure cannot be negative")

    if (df["monthly_charges"] < 0).any():
        raise ValueError("Monthly charges cannot be negative")

    if (df["support_calls"] < 0).any():
        raise ValueError("Support calls cannot be negative")

    # Check contract categories
    if not set(df["contract"].unique()).issubset(
        {"Monthly", "One Year", "Two Year"}
    ):
        raise ValueError("Unexpected contract type")

    # Check internet service categories
    if not set(df["internet_service"].unique()).issubset(
        {"DSL", "Fiber", "None"}
    ):
        raise ValueError("Unexpected internet service")

    # Check that both churn classes exist
    if df["churn"].nunique() != 2:
        raise ValueError(
            "Target must contain both churn classes"
        )

    print(f"Data validation passed: {len(df)} rows")

    return df


if __name__ == "__main__":
    validate_data()

