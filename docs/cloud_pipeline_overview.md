# Cloud Pipeline Overview

## Purpose

This extension shows how a trained churn model can be used inside a small analytics workflow. It separates data storage, data preparation, prediction, reporting, and notification so each part can be checked independently.

## What happens

```text
1. The public IBM Telco dataset is loaded into Snowflake.
2. SQL keeps a raw layer and creates a curated, analysis-ready view.
3. Python checks the curated data and compares it with the existing cleaning path.
4. The saved Gradient Boosting model scores the customers.
5. A CSV report stores probabilities, labels, threshold, and historical outcome.
6. Power BI displays the main churn patterns and risk distribution.
7. OneDrive and Power Automate send a generic Gmail notification when the report changes.
```

## What this does not do

- It does not retrain the model.
- It does not automatically synchronize a local VS Code file with OneDrive.
- It does not connect Power BI directly to live production customer data.
- It does not prove that a contract type, payment method, or other segment causes churn.

## Main outputs

The cloud scoring script creates `reports/cloud_predictions.csv` with fields including:

| Field | Meaning |
|---|---|
| `customerID` | Stable identifier from the learning dataset. |
| `churn_probability` | Model-estimated risk between 0 and 1. |
| `predicted_churn` | Label created from the selected threshold. |
| `decision_threshold` | Threshold used to create the label. |
| `actual_churn` | Historical outcome included for evaluation in this dataset. |

The Power BI report is a decision-support view of this generated file. Its results describe patterns in the dataset; they are not causal conclusions.

## Security and cost boundaries

- Snowflake credentials stay in a local `.env` file.
- The real `.env`, private keys, and generated prediction CSV are ignored by Git.
- The Snowflake warehouse should use automatic suspension.
- Only the public learning dataset is used.
- The notification contains a generic update message, not customer records or credentials.

## Reproduce the cloud scoring step

See the [Snowflake CLI guide](snowflake_cli_guide.md) for the connection check, curated-view check, equivalence check, and scoring command.

The [Power Automate guide](power_automate_flow.md) explains the separate notification step and its manual OneDrive upload boundary.
