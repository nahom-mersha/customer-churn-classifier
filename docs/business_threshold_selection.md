# Business-Cost-Aware Threshold Selection

## Objective

The model outputs churn probabilities, but the business needs a decision rule:

```text
if churn_probability >= threshold:
    contact customer
else:
    do not contact customer
```

The goal of this step is to choose a classification threshold based on an explicit business-cost model instead of using the default threshold of `0.50`.

## Cost assumptions

This project uses a simplified illustrative cost model:

| Outcome | Meaning | Net value |
|---|---|---:|
| True positive | Contact a customer who would churn | +€180 |
| False positive | Contact a customer who would not churn | -€20 |
| False negative | Miss a customer who would churn | -€180 |
| True negative | Correctly leave a non-churner alone | €0 |

The assumptions are:

- Retention offer cost: €20
- Value of successfully retaining a real churner: €200
- Net true-positive value: €200 - €20 = €180
- Missed opportunity cost for a false negative: -€180

These values are illustrative. They are not universal business facts.

## Net value formula

For each threshold, the business value is calculated as:

```text
net_value = 180 * TP - 20 * FP - 180 * FN
```

True negatives are omitted because their value is assumed to be €0.

## Validation method

Threshold selection was performed using out-of-fold training probabilities, not the held-out test set.

This avoids selecting a threshold directly on the final test set. Each out-of-fold probability is produced by a model that did not train on that specific customer.

The threshold sweep evaluated candidate thresholds from `0.10` to `0.90` in steps of `0.05`.

## Results

### Logistic Regression

Best threshold:

```text
threshold = 0.10
TP = 1408
FP = 2110
FN = 87
TN = 2029
net_value = €195,580
```

At the default threshold of `0.50`:

```text
TP = 812
FP = 432
FN = 683
TN = 3707
net_value = €14,580
```

### Gradient Boosting

Best threshold:

```text
threshold = 0.10
TP = 1420
FP = 2146
FN = 75
TN = 1993
net_value = €199,180
```

At the default threshold of `0.50`:

```text
TP = 789
FP = 384
FN = 706
TN = 3755
net_value = €7,260
```

## Decision

The selected business-cost-aware decision policy is:

```text
Selected model: Gradient Boosting
Selected threshold: 0.10
```

This was selected because Gradient Boosting achieved the highest out-of-fold net value under the stated cost assumptions:

```text
Gradient Boosting net value: €199,180
Logistic Regression net value: €195,580
Difference: €3,600
```

## Interpretation

The selected threshold is much lower than `0.50` because the cost model makes false negatives much more expensive than false positives.

At threshold `0.10`, Gradient Boosting catches many more churners:

```text
TP = 1420
FN = 75
```

At threshold `0.50`, it misses many more churners:

```text
TP = 789
FN = 706
```

So the lower threshold deliberately accepts more false positives in exchange for far fewer false negatives.

This is an aggressive retention policy:

```text
if churn_probability >= 0.10:
    contact customer
```

## Limitation

This threshold is not universally optimal. It is optimal only for the simplified cost assumptions used in this project.

If the retention offer cost increased, or if the value of retaining a churner decreased, the best threshold could move higher.

## Final held-out test evaluation

After selecting the model and threshold using out-of-fold training probabilities, the final Gradient Boosting pipeline was trained on the training split and evaluated once on the held-out test set.

The selected decision policy was not changed after this evaluation:

```text
Selected model: Gradient Boosting
Selected threshold: 0.10
```

Final held-out test results:

| Metric | Value |
|---|---:|
| TN | 492 |
| FP | 543 |
| FN | 17 |
| TP | 357 |
| Accuracy | 0.6026 |
| Precision | 0.3967 |
| Recall | 0.9545 |
| F1 | 0.5604 |
| ROC-AUC | 0.8467 |
| Average Precision | 0.6684 |
| Brier score | 0.1349 |
| Net value | €50,340 |

## Final test interpretation

The selected threshold gives very high churn recall on the held-out test set:

```text
Recall = 0.9545
FN = 17
TP = 357
```

This means the model catches most customers who actually churn.

The trade-off is a large number of false positives:

```text
FP = 543
Precision = 0.3967
```

So many contacted customers would not have churned. This is expected because the selected threshold is low and the cost model treats missed churners as much more expensive than unnecessary retention contacts.

This final test result was used only for reporting. The threshold was not retuned after seeing the held-out test performance.