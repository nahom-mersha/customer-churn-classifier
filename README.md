# Customer Churn Classifier

A machine-learning project that predicts whether a customer is likely to leave a service.

This project implements a customer churn classification workflow using a public telecom dataset. It includes from-scratch logistic regression with NumPy, professional scikit-learn model comparisons, probability-focused evaluation, business-cost-aware threshold selection, and a reproducible final model training workflow.

## Project Status

Steps 1-14 are complete:

- problem definition;
- dataset selection;
- reproducible project and data structure;
- data cleaning and validation;
- data-quality reporting;
- leakage-safe preprocessing;
- majority-class baseline evaluation;
- from-scratch logistic regression with NumPy;
- numerical gradient checking;
- from-scratch model threshold evaluation;
- reusable scikit-learn pipelines;
- professional model comparison;
- probability quality, calibration, and error analysis;
- business-cost-aware threshold selection;
- configuration-driven final model training and saved metadata.

Current step: Step 15, batch prediction and API endpoint.

## Dataset

This project uses the classic IBM Telco Customer Churn dataset.

The raw dataset is not committed to the repository. Download it from Kaggle and place it at:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Dataset details and cleaning decisions are documented in:

```text
data/dataset.md
```

The full project roadmap is documented in:

```text
docs/roadmap.md
```

## Preprocessing

Preprocessing is fitted only on the training set. The test set is transformed using the fitted training preprocessor, so imputation values, scaling values, and one-hot encoded categories are not learned from the test data.

The preprocessing pipeline:

- imputes missing numeric values with the training-set median;
- scales numeric features;
- fills missing categorical values with `missing`;
- one-hot encodes categorical features;
- ignores unknown categories during future transforms.

The saved final model keeps preprocessing and the classifier together in one scikit-learn pipeline, so future batch or API predictions use the same transformations as training.

## Final Model

The selected final model is Gradient Boosting.

The selected business-cost-aware decision threshold is:

```text
0.10
```

This threshold was selected using out-of-fold training probabilities and an illustrative business cost model. It is intentionally lower than the default `0.50` threshold because the project assumes that missing a real churner is more expensive than contacting a customer who would not have churned.

The final held-out test results for the selected model and threshold are:

| Metric | Value |
|---|---:|
| Accuracy | 0.6026 |
| Precision | 0.3967 |
| Recall | 0.9545 |
| F1 | 0.5604 |
| ROC-AUC | 0.8467 |
| Average Precision | 0.6684 |
| Brier score | 0.1349 |
| Net value | €50,340 |

The final threshold decision is documented in:

```text
docs/business_threshold_selection.md
```

## Final Model Training

The final model training configuration is stored in:

```text
configs/final_model.yaml
```

To train the final preprocessing-and-model pipeline and generate saved artifacts, run:

```bash
python scripts/train_final_model.py
```

This command creates:

```text
models/customer_churn_gradient_boosting.joblib
models/customer_churn_gradient_boosting_metadata.json
```

The `.joblib` model artifact is generated locally and ignored by Git. The metadata JSON is committed as a lightweight record of the selected model, threshold, metrics, feature list, and artifact paths.

## Development Tools

- `src` layout
- `pytest`
- Ruff
- Logging
- YAML configuration
- GitHub Actions
- Docker

## Limitations

This is a learning project using a public churn dataset. The selected threshold and business-cost assumptions are illustrative and should not be treated as universal business rules.

Predictions are intended for decision support, not automatic customer treatment.