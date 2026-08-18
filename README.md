# Customer Churn Classifier

A machine-learning project that predicts whether a customer is likely to leave a service.

This project implements a customer churn classification workflow using a public telecom dataset. It will include from-scratch logistic regression with NumPy, professional scikit-learn model comparisons, probability-focused evaluation, and deployment-ready prediction workflows.

## Project Status

Steps 1-5 are complete:

- problem definition;
- dataset selection;
- data cleaning;
- data-quality reporting;
- leakage-safe preprocessing.

Current step: Step 6, majority-class baseline evaluation.

## Dataset

This project uses the classic IBM Telco Customer Churn dataset.

The raw dataset is not committed to the repository. Download it from Kaggle and place it at:

`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`

Dataset details and cleaning decisions are documented in:

`data/dataset.md`

The full project roadmap is documented in:

`docs/roadmap.md`

## Preprocessing

Preprocessing is fitted only on the training set. The test set is transformed using the fitted training preprocessor, so imputation values, scaling values, and one-hot encoded categories are not learned from the test data.

The preprocessing pipeline:

- imputes missing numeric values with the training-set median;
- scales numeric features;
- fills missing categorical values with `missing`;
- one-hot encodes categorical features;
- ignores unknown categories during future transforms.

## Development Tools

- `src` layout
- `pytest`
- Ruff
- Logging
- YAML configuration
- GitHub Actions
- Docker