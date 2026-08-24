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

## Error analysis at threshold 0.50

After analyzing ranking quality, probability quality, and calibration, the next step was to inspect the model's mistakes.

The error analysis used the Gradient Boosting model because it is the current primary candidate. A temporary classification threshold of `0.50` was used.

This threshold is not the final business threshold. It is only a starting point for understanding the model's false positives and false negatives.

## Error counts

At threshold `0.50`, the Gradient Boosting model produced the following results on the out-of-fold training predictions:

| Error type | Count |
|---|---:|
| Correct | 4544 |
| False negative | 706 |
| False positive | 384 |

The model made more false negatives than false positives.

For this churn problem:

- a false positive means the model predicted churn, but the customer actually stayed;
- a false negative means the model predicted no churn, but the customer actually churned.

This is important because false negatives represent missed churners. In a retention use case, these are customers the business may fail to contact before they leave.

## Numeric feature patterns

| Error type | Average tenure | Average MonthlyCharges | Average TotalCharges |
|---|---:|---:|---:|
| Correct | 34.87 | 63.19 | 2399.35 |
| False negative | 28.49 | 68.84 | 2356.55 |
| False positive | 11.66 | 78.32 | 1060.67 |

False positives had the shortest average tenure and the highest average monthly charges.

This suggests that the model strongly associates short customer tenure and high monthly charges with churn risk. Some customers with these high-risk signals still stayed, so they became false positives.

False negatives had intermediate values. They had shorter tenure and higher monthly charges than the correctly classified group, but they were less extreme than the false positives. This may explain why some actual churners did not cross the `0.50` decision threshold.

## Contract patterns

| Error type | Month-to-month | One year | Two year |
|---|---:|---:|---:|
| Correct | 47.99% | 22.95% | 29.05% |
| False negative | 76.06% | 18.41% | 5.52% |
| False positive | 100.00% | 0.00% | 0.00% |

All false positives were month-to-month customers.

This makes sense because month-to-month contracts are usually more flexible and are often associated with higher churn risk. The model appears to treat month-to-month contracts as a strong churn signal.

False negatives were also mostly month-to-month customers, but not as extremely as false positives.

## Internet service patterns

| Error type | Fiber optic | DSL | No internet service |
|---|---:|---:|---:|
| Correct | 39.81% | 35.37% | 24.82% |
| False negative | 51.27% | 36.54% | 12.18% |
| False positive | 81.25% | 18.75% | 0.00% |

False positives were heavily concentrated among Fiber optic customers.

This suggests that Fiber optic service is another strong churn-risk signal for the model. Many false positives looked risky because they had Fiber optic service, but they did not actually churn.

False negatives also had a higher Fiber optic share than the correctly classified group, but again the pattern was less extreme than for false positives.

## Payment method patterns

| Error type | Electronic check | Mailed check | Credit card automatic | Bank transfer automatic |
|---|---:|---:|---:|---:|
| Correct | 29.42% | 23.88% | 22.84% | 23.86% |
| False negative | 37.82% | 22.10% | 20.26% | 19.83% |
| False positive | 74.74% | 11.72% | 8.33% | 5.21% |

False positives were strongly concentrated among customers using Electronic check.

This suggests that Electronic check is another important churn-risk signal. The model often flagged Electronic check customers as likely churners, but many of those customers still stayed.

False negatives also had a higher Electronic check share than the correctly classified group, but the pattern was much less extreme than for false positives.

## Error analysis conclusion

The error analysis shows that the model's mistakes are not random.

False positives tend to look like very high-risk customers:

- very short tenure;
- high monthly charges;
- month-to-month contracts;
- Fiber optic internet service;
- Electronic check payment method.

These customers have many features commonly associated with churn, so the model flags them. Some of them still stay, which creates false positives.

False negatives are actual churners that the model missed. They also show some churn-risk signals, especially month-to-month contracts, Fiber optic service, and Electronic check payment, but their profile is generally less extreme than the false positives.

This suggests that the default `0.50` threshold may be too strict if the business wants to catch more churners. A lower threshold may reduce false negatives, but it would likely increase false positives.

The final decision threshold should therefore be selected later using a business-cost-aware approach, rather than assuming that `0.50` is the best threshold.