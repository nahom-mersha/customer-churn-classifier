# Probability Quality and Calibration Analysis

## Purpose

This document summarizes analyzing ranking quality, probability quality, and calibration for the two main candidate models:

- Logistic Regression
- Gradient Boosting

The goal is not only to ask which model performs better, but also whether the predicted churn probabilities are useful and trustworthy.

## Evaluation setup

The analysis used out-of-fold predictions on the training data.

This means each training customer received a predicted churn probability from a model that did not train on that exact customer.

The held-out test set was not used.

This keeps the probability analysis more honest while preserving the test set for final evaluation later.

## Metrics

The following metrics were calculated:

| Model | ROC-AUC | Average Precision | Brier score |
|---|---:|---:|---:|
| Logistic Regression | 0.8456 | 0.6583 | 0.1351 |
| Gradient Boosting | 0.8500 | 0.6686 | 0.1331 |

## Ranking quality

Ranking quality asks:

> Does the model generally rank real churners above non-churners?

This was measured using:

- ROC-AUC
- Average Precision
- ROC curve
- Precision-Recall curve

Gradient Boosting had slightly better ranking quality than Logistic Regression:

- higher ROC-AUC;
- higher Average Precision;
- slightly stronger Precision-Recall behavior.

However, the difference was small. Logistic Regression remains a useful reference model because it is simpler and easier to interpret.

## ROC curve interpretation

The ROC curve showed that both models separate churners from non-churners reasonably well.

Both models had ROC-AUC values around 0.85, meaning they are good at ranking actual churners above actual non-churners.

Gradient Boosting was slightly stronger, but the two curves were very close.
![ROC curve](../reports/figures/roc_curve.png)


## Precision-Recall curve interpretation

The Precision-Recall curve is especially important for this project because churn is the minority class.

It shows the trade-off between:

- catching more actual churners;
- keeping the flagged churn group accurate.

As recall increased, precision decreased. This is expected because catching more churners requires flagging more customers, which also increases false positives.

Gradient Boosting had a slightly higher Average Precision score, meaning it maintained slightly better precision while retrieving churners.
![Precision-Recall curve](../reports/figures/precision_recall_curve.png)

## Probability quality

Probability quality asks:

> Are the predicted probability values themselves trustworthy?

This was measured using:

- Brier score
- calibration curve / reliability diagram

Gradient Boosting had a slightly better Brier score:

```text
Logistic Regression Brier score: 0.1351
Gradient Boosting Brier score: 0.1331
```

Since lower Brier score is better, Gradient Boosting had slightly better overall probability quality.

## Calibration curve interpretation

The calibration curve compares predicted churn probabilities with actual observed churn rates.

A perfectly calibrated model would follow the diagonal line, where:

```text
predicted probability ≈ actual churn rate
```

Both Logistic Regression and Gradient Boosting stayed reasonably close to the diagonal.

This suggests that both models are fairly well calibrated before applying any explicit calibration method.

Small local deviations were visible:

- points below the diagonal mean the model is slightly overconfident;
- points above the diagonal mean the model is slightly underconfident.

Overall, neither model showed severe miscalibration.
![Calibration curve](../reports/figures/calibration_curve.png)
## Calibration decision

Explicit calibration with sigmoid or isotonic calibration will not be applied at this stage.

Reason:

- both models already appear reasonably close to the calibration diagonal;
- Gradient Boosting also has the slightly better Brier score;
- calibration can still be revisited later if the final business-cost threshold requires more precise probability estimates.

## Current conclusion

Gradient Boosting remains the primary candidate model.

Logistic Regression remains a reference candidate because its performance is close and it is easier to interpret.