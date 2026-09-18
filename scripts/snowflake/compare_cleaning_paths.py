import argparse
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal

from customer_churn_classifier.data import clean_churn_data
from customer_churn_classifier.snowflake_data import load_curated_data

BASE_COLUMNS = [
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


NUMERIC_COLUMNS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


def normalize_numeric_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert numeric columns to a consistent pandas numeric type."""
    normalized = dataframe.copy()

    for column in NUMERIC_COLUMNS:
        normalized[column] = pd.to_numeric(
            normalized[column],
            errors="coerce",
        ).astype("float64")

    normalized["MonthlyCharges"] = normalized["MonthlyCharges"].round(2)
    normalized["TotalCharges"] = normalized["TotalCharges"].round(2)

    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()

    # Load and clean the local CSV using the existing Python contract.
    local_raw = pd.read_csv(args.csv_path)
    local_cleaned = clean_churn_data(local_raw)
    local_cleaned = local_cleaned[BASE_COLUMNS].copy()

    # Load the curated data from Snowflake.
    snowflake_data = load_curated_data()
    snowflake_cleaned = snowflake_data[BASE_COLUMNS].copy()

    # Sort both datasets by the stable customer identifier.
    local_cleaned = local_cleaned.sort_values("customerID").reset_index(drop=True)
    snowflake_cleaned = snowflake_cleaned.sort_values("customerID").reset_index(
        drop=True
    )

    # Make numeric representations comparable across CSV, pandas, and Snowflake.
    local_cleaned = normalize_numeric_columns(local_cleaned)
    snowflake_cleaned = normalize_numeric_columns(snowflake_cleaned)

    # Verify that both cleaning paths produce equivalent data.
    assert_frame_equal(
        local_cleaned,
        snowflake_cleaned,
        check_dtype=False,
        check_exact=False,
    )

    print("Cleaning paths are equivalent.")
    print(f"Rows compared: {len(local_cleaned)}")
    print(f"Columns compared: {len(BASE_COLUMNS)}")
    print("Target distribution:")
    print(local_cleaned["Churn"].value_counts().sort_index())


if __name__ == "__main__":
    main()
