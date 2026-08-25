# Model Card — Customer Churn Classifier

## Model overview

This project is a customer churn classification system. Given customer information and service usage features, the model estimates the probability that a customer will churn.

The selected final model is a scikit-learn Gradient Boosting classifier wrapped in a preprocessing-and-model pipeline.

The saved pipeline includes both preprocessing and the classifier, so batch and API predictions use the same transformations as training.

## Intended use

The model is intended for learning and decision-support use.

Example intended use:

- A retention team reviews customers with high predicted churn risk.
- Customers above the selected threshold may be considered for a retention contact or offer.
- Predictions should support human decision-making, not automatically determine customer treatment.

## Non-use cases

This model should not be used to:

- automatically penalize, deny service to, or disadvantage customers;
- make real business decisions without validating costs, assumptions, and performance on current company data;
- infer sensitive personal characteristics;
- replace human review in retention decisions;
- claim universal churn behavior across companies or industries.

## Dataset

The project uses the IBM Telco Customer Churn dataset.

The dataset is a public learning dataset with customer service, account, and contract information. The target is whether a customer churned.

The raw dataset is not committed to the repository. It should be placed locally at:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Dataset documentation and cleaning decisions are stored in:

```text
data/dataset.md
```

## Target definition

The target variable is churn.

In the cleaned project dataset, churn is encoded as:

```text
0 = did not churn
1 = churned
```

The model predicts the probability of class `1`.

## Features

The final model metadata records the following input features:

```text
SeniorCitizen
tenure
MonthlyCharges
TotalCharges
gender
Partner
Dependents
PhoneService
MultipleLines
InternetService
OnlineSecurity
OnlineBackup
DeviceProtection
TechSupport
StreamingTV
StreamingMovies
Contract
PaperlessBilling
PaymentMethod
```

The model does not use `customerID` as a feature.

## Preprocessing

The preprocessing pipeline is fitted on the training split only.

The preprocessing includes:

- median imputation for numerical features;
- scaling for numerical features;
- most-frequent or missing-value handling for categorical features;
- one-hot encoding for categorical features;
- ignoring unknown categories at inference time.

The preprocessing and classifier are saved together in one pipeline artifact.

## Model selection

The project compared multiple approaches, including:

- majority-class baseline;
- from-scratch logistic regression with NumPy;
- scikit-learn logistic regression;
- K-nearest neighbours;
- decision tree;
- random forest;
- gradient boosting.

Gradient Boosting was selected as the final model because it performed strongly in the project’s probability/ranking evaluation and gave the best business net value under the stated threshold assumptions.

## Selected threshold

The selected business-cost-aware threshold is:

```text
0.10
```

Decision policy:

```text
if churn_probability >= 0.10:
    predicted_churn = 1
else:
    predicted_churn = 0
```

This threshold is lower than the default `0.50` because the project assumes that missing a real churner is more expensive than contacting a customer who would not have churned.

## Business cost assumptions

The project uses an illustrative cost model:

| Outcome | Meaning | Net value |
|---|---|---:|
| True positive | Contact a customer who would churn | +€180 |
| False positive | Contact a customer who would not churn | -€20 |
| False negative | Miss a customer who would churn | -€180 |
| True negative | Correctly leave a non-churner alone | €0 |

The net-value formula is:

```text
net_value = 180 * TP - 20 * FP - 180 * FN
```

These values are illustrative assumptions for this learning project. They are not universal business facts.

## Final held-out test performance

The final selected model and threshold were evaluated once on the held-out test set.

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

Confusion matrix counts:

| Count | Value |
|---|---:|
| True negatives | 492 |
| False positives | 543 |
| False negatives | 17 |
| True positives | 357 |

## Interpretation

The selected threshold produces very high churn recall.

This means the model catches most customers who churn, but it also produces many false positives. That trade-off is intentional under the project’s cost assumptions.

The model is optimized for a retention scenario where missing churners is considered more costly than contacting extra customers.

## Probability quality

The final model reports probabilities, not only labels.

The Brier score on the held-out test set was:

```text
0.1349
```

This gives a measure of probability quality. Lower is better.

The model’s probabilities should still be treated carefully. A high ranking score does not automatically mean the probabilities are perfectly calibrated for real-world decision-making.

## Inference workflows

The project supports two inference workflows.

Batch prediction:

```bash
python scripts/predict_batch.py --input data/processed/telco_churn_clean.csv --output reports/batch_predictions.csv
```

API prediction:

```bash
uvicorn customer_churn_classifier.api:app --reload
```

Interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

Both workflows load the saved model and metadata from the configured artifact paths.

## Artifacts

Configuration:

```text
configs/final_model.yaml
```

Generated model artifact:

```text
models/customer_churn_gradient_boosting.joblib
```

Committed metadata artifact:

```text
models/customer_churn_gradient_boosting_metadata.json
```

The `.joblib` file is generated locally and ignored by Git. The metadata JSON is committed as a lightweight record of the selected model, threshold, metrics, feature list, and artifact paths.

## Limitations

This is a learning project using a public churn dataset.

Important limitations:

- The dataset is not evidence that the model would perform well for a real company.
- The business cost assumptions are illustrative.
- The threshold is not universally optimal.
- The final model should be revalidated before any real-world deployment.
- The model may perform differently across customer segments.
- The dataset may not include all relevant churn drivers.
- The API and batch workflows are local project implementations, not hardened production services.

## Ethical and practical risks

Potential risks include:

- over-contacting customers who would not have churned;
- missing some customers who would churn;
- relying too heavily on model outputs without human review;
- using a threshold that does not match real business costs;
- assuming public learning-data performance transfers to a real company.

Predictions should be treated as decision support only.

## Recommended future improvements

Possible improvements include:

- more detailed segment-level error analysis;
- stronger calibration analysis;
- monitoring for data drift;
- clearer production retraining policy;
- API authentication and deployment hardening;
- stronger input validation with explicit allowed categories;
- more tests for invalid API and CSV inputs.
