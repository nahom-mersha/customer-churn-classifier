# Dataset

## Source

This project uses the classic IBM Telco Customer Churn dataset, commonly available on Kaggle as `blastchar/telco-customer-churn`.

Dataset link: https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Description

The dataset contains 7,043 fictional telecom customer records. Each row represents one customer, with information about their contract, billing, tenure, services, and whether they churned.

## Target

The target column is `Churn`, which indicates whether the customer left the company.

## Prediction-Time Features

The model will use only customer information that would reasonably be available at prediction time, such as tenure, contract type, monthly charges, payment method, and subscribed services.

## Why This Dataset Fits

This dataset is useful for learning binary classification, categorical preprocessing, class imbalance, probability estimation, and model evaluation.

## Limitations

This is a fictional learning dataset and does not provide a perfect real-world temporal prediction window. Therefore, results should be interpreted as a learning exercise, not as proof of production telecom performance.

## Leakage Warning

We will not use columns that directly reveal churn after the fact, such as churn reason, churn score, churn category, customer status, cancellation date, or any field created after the customer already churned.

## Initial Inspection

The raw dataset has 7,043 rows and 21 columns. The target column is `Churn`, with values `Yes` and `No`.

Most columns are categorical strings. The main numeric columns are:

- `SeniorCitizen`
- `tenure`
- `MonthlyCharges`

The `TotalCharges` column is read as a string because it contains 11 blank values. This must be cleaned before modeling by converting blank values to missing values and then converting the column to numeric.

The `customerID` column is an identifier and will not be used as a model feature.

## Cleaning Decisions

- Drop `customerID` from the model features because it is only an identifier.
- Convert `TotalCharges` from string to numeric.
- Treat the 11 blank `TotalCharges` values as missing values.
- Encode `Churn` as the binary target: `Yes = 1`, `No = 0`.

The dataset CSV is not committed to the repository. Download it from Kaggle and place it at:

`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`