# Snowflake CLI Guide

This guide explains how to run the Snowflake-related command-line scripts in the customer churn project.

The scripts follow this flow:

```text
Snowflake curated view -> Python DataFrame -> existing churn model -> predictions CSV
```

## Before running a script

From the project root, activate the virtual environment and make sure the local `.env` file exists:

```powershell
.\.venv\Scripts\Activate.ps1
```

The `.env` file contains the Snowflake connection details. It is local-only and must never be committed or pasted into logs:

```text
SNOWFLAKE_ACCOUNT=your_account_identifier
SNOWFLAKE_USER=your_username
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=CHURN_WH
SNOWFLAKE_DATABASE=CHURN_DB
SNOWFLAKE_SCHEMA=ANALYTICS
SNOWFLAKE_ROLE=ACCOUNTADMIN
```

Run commands from the repository root so that imports and relative model paths work correctly.

## 1. Test the Snowflake connection

```powershell
python scripts\snowflake\test_snowflake_connection.py
```

This checks that Python can authenticate and reach Snowflake. It prints the current user, role, database, schema, and warehouse. It does not change project data.

Expected result:

```text
Snowflake connection succeeded.
```

## 2. Read the curated view

```powershell
python scripts\snowflake\read_curated_view.py
```

This runs a `SELECT` query against `ANALYTICS.TELCO_CHURN_CURATED` and loads the result into pandas. It is a read-only inspection step.

The full dataset should contain 7,043 rows and 23 columns, including the two analysis fields `TENURE_BAND` and `MONTHLY_CHARGE_BAND`.

## 3. Compare Python and Snowflake cleaning

```powershell
python scripts\snowflake\compare_cleaning_paths.py path\to\Telco-Customer-Churn.csv
```

This compares the existing local `clean_churn_data()` function with the Snowflake curated view loaded through Python. Both datasets are sorted by `customerID`, numeric columns are normalized, and the values are compared.

Expected result:

```text
Cleaning paths are equivalent.
Rows compared: 7043
Columns compared: 21
```

## 4. Run the cloud-to-model pipeline

```powershell
python scripts\snowflake\run_cloud_pipeline.py --output reports\cloud_predictions.csv
```

This script loads the curated Snowflake view, writes a temporary CSV because the existing batch predictor accepts CSV input, reuses the saved model and metadata, creates churn probabilities and labels, and saves the final report locally.

The generated report contains the original customer fields plus:

| Column | Meaning |
| --- | --- |
| `actual_churn` | Known historical outcome from the dataset: 0 or 1 |
| `churn_probability` | Probability assigned by the saved model |
| `predicted_churn` | Model decision: 0 or 1 |
| `decision_threshold` | Probability threshold used to create the prediction |

`reports\cloud_predictions.csv` is a generated output and should remain ignored by Git.

## Existing local batch CLI

The cloud pipeline reuses the existing local prediction command rather than duplicating model logic:

```powershell
python scripts\predict_batch.py `
  --input data\processed\input.csv `
  --output reports\predictions.csv
```

Reusable logic stays in `src/customer_churn_classifier/`; files under `scripts/` are thin command-line entry points.

## Troubleshooting

### `ModuleNotFoundError`

Confirm that the virtual environment is active and dependencies are installed:

```powershell
uv sync
```

### Authentication or connection errors

Check the values in `.env`, especially the account identifier, username, password, role, and region/account identifier. Never add the password to a Python file.

### scikit-learn version warning

`InconsistentVersionWarning` means the saved model was created with a slightly different scikit-learn version than the current environment. It is a warning, not a pipeline failure. The pipeline can still complete; matching the training version is a later maintenance task.

### No rows returned

Check that the Snowflake objects exist and that the curated view was created and loaded:

```sql
SELECT COUNT(*) FROM CHURN_DB.ANALYTICS.TELCO_CHURN_CURATED;
```

## Verification checklist

- [ ] `.env` exists locally and is ignored by Git.
- [ ] Connection test succeeds.
- [ ] Curated view returns 7,043 rows.
- [ ] Cleaning comparison reports equivalence.
- [ ] Cloud pipeline produces `reports\cloud_predictions.csv`.
- [ ] Existing automated tests still pass.
