from pathlib import Path

import pandas as pd

RAW_DATA_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
PROCESSED_DATA_PATH = Path("data/processed/telco_churn_clean.csv")

TARGET_COLUMN = "Churn"
ID_COLUMN = "customerID"

EXPECTED_COLUMNS = [
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


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Telco Customer Churn dataset."""
    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame) -> None:
    """Check that the raw dataset has the expected columns."""
    missing_columns = set(EXPECTED_COLUMNS) - set(df.columns)

    if missing_columns:
        raise ValueError(f"Missing expected columns: {sorted(missing_columns)}")


def clean_churn_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw Telco Customer Churn dataset."""
    validate_columns(df)

    cleaned = df.copy()

    cleaned["TotalCharges"] = cleaned["TotalCharges"].replace(" ", pd.NA)
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"])

    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].map(
        {
            "No": 0,
            "Yes": 1,
        }
    )

    return cleaned


def save_processed_data(
    df: pd.DataFrame,
    path: Path = PROCESSED_DATA_PATH,
) -> None:
    """Save the cleaned dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_clean_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load and clean the Telco Customer Churn dataset."""
    raw_df = load_raw_data(path)
    return clean_churn_data(raw_df)


def build_processed_dataset(
    raw_path: Path = RAW_DATA_PATH,
    processed_path: Path = PROCESSED_DATA_PATH,
) -> pd.DataFrame:
    """Create and save the cleaned dataset."""
    cleaned = load_clean_data(raw_path)
    save_processed_data(cleaned, processed_path)
    return cleaned
