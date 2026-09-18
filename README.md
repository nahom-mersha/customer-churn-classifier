# Customer Churn Classifier

An end-to-end learning project that estimates which telecom customers may be at risk of leaving and presents the results in a simple analytics workflow.

The project uses the public IBM Telco Customer Churn dataset. It combines data validation, machine learning, batch and API prediction, Snowflake analytics, Power BI reporting, and a small email notification flow.

## What the project is for

The system helps a retention team identify groups of customers that may deserve human review. It produces risk estimates; it does not prove that a customer will leave and it should not make automatic decisions about people.

## How the complete workflow fits together

```text
Public telecom data
        ↓
Snowflake raw table and curated SQL view
        ↓
Python validation and existing saved model
        ↓
cloud_predictions.csv
        ↓
Power BI dashboard
        ↓
OneDrive update → Power Automate → Gmail notification
```

The cloud step performs inference only. It applies the already-trained model to curated Snowflake data; it does not retrain the model.

## Main results

The selected model is Gradient Boosting. On the held-out test set:

| Metric | Result | Plain meaning |
|---|---:|---|
| ROC-AUC | 0.8467 | The model ranks higher-risk customers above lower-risk customers reasonably well. |
| Recall | 0.9545 | The selected threshold found most churners in the test data. |
| Precision | 0.3967 | Many flagged customers would not have churned, so human review remains necessary. |
| Decision threshold | 0.10 | The policy favors finding more possible churners over reducing follow-up volume. |

These results come from a fictional learning dataset and are not evidence of performance for a real company.

## What is included

- leakage-safe cleaning and preprocessing;
- logistic regression implemented from scratch with NumPy;
- scikit-learn model comparison and probability evaluation;
- business-cost-aware threshold selection;
- saved model metadata, batch prediction, and FastAPI inference;
- Snowflake raw and curated data layers with version-controlled SQL;
- Python validation and cloud-to-model scoring;
- Power BI dashboard: [`reports/customer_churn_dashboard.pbix`](reports/customer_churn_dashboard.pbix);
- OneDrive and Power Automate notification prototype using Gmail.

## Run the core project locally

Place the public dataset at:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Then install and run the main workflow:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
python scripts/build_dataset.py
python scripts/train_final_model.py
python scripts/predict_batch.py \
  --input data/processed/telco_churn_clean.csv \
  --output reports/batch_predictions.csv
```

Run the API with:

```bash
uvicorn customer_churn_classifier.api:app --reload
```

Quality checks:

```bash
ruff check .
pytest
```

## Run the cloud extension

Create a local `.env` from [`.env.example`](.env.example) and fill in the Snowflake connection values. Never commit the real `.env` file or credentials.

The SQL setup files are in [`sql/`](sql/). After Snowflake is prepared, the main cloud scoring command is:

```bash
python scripts/snowflake/run_cloud_pipeline.py \
  --output reports/cloud_predictions.csv
```

The generated CSV is the input for the Power BI report. The detailed command and validation steps are in [`docs/snowflake_cli_guide.md`](docs/snowflake_cli_guide.md).

## Documentation

- [Cloud pipeline overview](docs/cloud_pipeline_overview.md) — plain-language architecture and boundaries.
- [Snowflake CLI guide](docs/snowflake_cli_guide.md) — setup, commands, and checks.
- [Power Automate flow](docs/power_automate_flow.md) — OneDrive trigger, Gmail action, and limitation.
- [Core project roadmap](docs/roadmap.md) — original machine-learning scope and completion criteria.

## Limitations and responsible use

- The dataset is public and fictional.
- The business costs used for threshold selection are illustrative assumptions.
- A prediction is a risk estimate, not a guarantee.
- The model must be revalidated on current company data before real use.
- Predictions should support human review, not automatically penalize or disadvantage customers.
- The cloud workflow is a learning-scale portfolio implementation, not a production system with scheduling, monitoring, or live customer-data controls.

## Technical analysis

Detailed model-development notes are organized under [`docs/technical/`](docs/technical/):

- [Evaluation strategy](docs/technical/evaluation_strategy.md)
- [Model comparison](docs/technical/model_comparison.md)
- [Probability quality analysis](docs/technical/probability_quality_analysis.md)
- [Scratch logistic regression evaluation](docs/technical/scratch_logistic_evaluation.md)

## Learning notes

[Project 3 — Customer Churn Classifier AI Notes](https://github.com/nahom-mersha/ai-notes/tree/main/Project%203%20-%20Customer%20Churn%20Classifier)

This is an AI-assisted learning project. I used ChatGPT to help generate and explain code, then reviewed the implementation, ran tests, explored the underlying concepts, and documented what I learned.
