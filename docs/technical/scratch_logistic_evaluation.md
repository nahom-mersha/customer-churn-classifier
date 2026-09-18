# NumPy Logistic Regression Evaluation

The from-scratch logistic regression model was evaluated on the fixed held-out test set.

Training loss decreased from `0.6931` to `0.4144`, indicating that gradient descent learned a useful decision boundary.

## Threshold comparison

| Threshold | Precision | Recall | F1 |
|---|---:|---:|---:|
| 0.30 | 0.5246 | 0.7406 | 0.6142 |
| 0.50 | 0.6401 | 0.5374 | 0.5843 |
| 0.70 | 0.7292 | 0.1872 | 0.2979 |

Lowering the classification threshold increased the number of customers predicted as churners. This reduced false negatives and improved recall, but increased false positives and reduced precision.

Raising the threshold had the opposite effect: fewer customers were predicted as churners, which increased precision but caused many more real churners to be missed.

The `0.30` threshold produced the highest F1 score among the three tested thresholds, but it is **not selected as the final operating threshold** because the held-out test set should not be used repeatedly for threshold tuning.

The raw churn probabilities were saved to:

`reports/logistic_numpy_test_probabilities.csv`

These probabilities will be reused later for calibration analysis and business-cost-aware threshold selection.

### Why the test set was used here

The held-out test set was used to check whether the from-scratch NumPy logistic regression model generalizes to customers it was not trained on.

The thresholds `0.30`, `0.50`, and `0.70` were compared on this test set only as a small diagnostic experiment to demonstrate how changing the classification threshold affects precision, recall, false positives, and false negatives.

This experiment was **not used to select the final operating threshold**. In particular, although `0.30` produced the highest F1 score among these three thresholds, selecting it for that reason would allow test-set performance to influence a model-development decision.

The final threshold will instead be selected using training/validation data and explicit business-cost assumptions. The held-out test set can then be used for the final evaluation of the completed decision policy.