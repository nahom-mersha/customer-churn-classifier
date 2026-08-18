from pathlib import Path

import pandas as pd

from customer_churn_classifier.data import PROCESSED_DATA_PATH, TARGET_COLUMN

REPORT_PATH = Path("reports/data_quality_report.md")


def summarize_shape(df: pd.DataFrame) -> str:
    return f"- Rows: {df.shape[0]}\n- Columns: {df.shape[1]}"


def summarize_target(df: pd.DataFrame) -> str:
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    percentages = df[TARGET_COLUMN].value_counts(normalize=True).sort_index() * 100

    lines = []
    for label, count in counts.items():
        percentage = percentages[label]
        lines.append(f"- `{label}`: {count} rows ({percentage:.2f}%)")

    return "\n".join(lines)


def summarize_missing_values(df: pd.DataFrame) -> str:
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if missing.empty:
        return "No missing values found."

    return "\n".join(f"- `{column}`: {count}" for column, count in missing.items())


def summarize_duplicates(df: pd.DataFrame) -> str:
    duplicate_count = df.duplicated().sum()
    return f"- Duplicate rows: {duplicate_count}"


def summarize_dtypes(df: pd.DataFrame) -> str:
    return "\n".join(f"- `{column}`: `{dtype}`" for column, dtype in df.dtypes.items())


def summarize_numeric_ranges(df: pd.DataFrame) -> str:
    numeric_columns = df.select_dtypes(include="number").columns

    lines = []
    for column in numeric_columns:
        lines.append(
            f"- `{column}`: min={df[column].min()}, "
            f"max={df[column].max()}, "
            f"mean={df[column].mean():.2f}"
        )

    return "\n".join(lines)


def summarize_categorical_values(df: pd.DataFrame, max_values: int = 10) -> str:
    categorical_columns = df.select_dtypes(include="object").columns

    lines = []
    for column in categorical_columns:
        values = df[column].value_counts(dropna=False).head(max_values)
        formatted_values = ", ".join(
            f"{value} ({count})" for value, count in values.items()
        )
        lines.append(f"- `{column}`: {formatted_values}")

    return "\n".join(lines)


def summarize_validation_checks(df: pd.DataFrame) -> str:
    checks = {
        "target_is_binary_0_1": set(df[TARGET_COLUMN].dropna().unique()) <= {0, 1},
        "tenure_non_negative": (df["tenure"] >= 0).all(),
        "monthly_charges_non_negative": (df["MonthlyCharges"] >= 0).all(),
        "total_charges_non_negative_or_missing": (
            df["TotalCharges"].dropna() >= 0
        ).all(),
    }

    return "\n".join(
        f"- `{name}`: {'PASS' if passed else 'FAIL'}" for name, passed in checks.items()
    )


def build_data_quality_report(df: pd.DataFrame) -> str:
    return f"""# Data Quality Report

## Shape

{summarize_shape(df)}

## Target Distribution

`{TARGET_COLUMN}` uses `0 = No churn` and `1 = Churn`.

{summarize_target(df)}

## Missing Values

{summarize_missing_values(df)}

## Duplicate Rows

{summarize_duplicates(df)}

## Column Data Types

{summarize_dtypes(df)}

## Numeric Ranges

{summarize_numeric_ranges(df)}

## Categorical Values

{summarize_categorical_values(df)}

## Validation Checks

{summarize_validation_checks(df)}

## Cleaning Decisions

- `customerID` is kept for reference, but not used as a model feature.
- `TotalCharges` blank strings were converted to missing numeric values.
- `Churn` was encoded as `No = 0` and `Yes = 1`.
- Missing `TotalCharges` values are retained for pipeline imputation.
"""


def save_data_quality_report(
    df: pd.DataFrame,
    path: Path = REPORT_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_data_quality_report(df), encoding="utf-8")


def load_processed_data(path: Path = PROCESSED_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)
