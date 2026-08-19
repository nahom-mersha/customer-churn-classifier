# Evaluation Strategy

## Model comparison strategy

The models will not be compared using accuracy alone.

The majority-class baseline achieved **73.46% accuracy while identifying 0% of churners**. This shows that accuracy can be misleading when the positive class is the minority class.

The main metric for comparing models will be **PR-AUC (Precision-Recall Area Under the Curve)**.

PR-AUC is appropriate for this problem because churn is the minority class and the main goal is to identify customers who are likely to churn. It evaluates the trade-off between precision and recall across different classification thresholds.

The following metrics will also be reported:

* **Recall** — the proportion of actual churners correctly identified.
* **Precision** — the proportion of predicted churners who actually churn.
* **F1 score** — a combined measure of precision and recall.
* **ROC-AUC** — measures how well the model ranks churners above non-churners across thresholds.
* **Accuracy** — reported for context, but not used as the main model-selection metric.

### Decision

The best candidate model will primarily be selected based on **PR-AUC**, while also considering recall, precision, F1 score, ROC-AUC, and the overall precision-recall trade-off.

---

## Threshold-selection strategy

The final classification threshold will not automatically be fixed at `0.50`.

Because the purpose of the model is to identify customers at risk of churn, **recall is treated as an important business constraint**. Missing too many actual churners would reduce the usefulness of the model for a retention team.

At the same time, maximizing recall alone would not be enough. A model that predicts every customer as a churner would achieve 100% recall but would also generate many false positives and unnecessary retention actions.

### Threshold-selection rule

The final operating threshold will be:

> **The threshold that achieves the highest precision while maintaining churn recall of at least 80%.**

This rule requires the model to identify at least 80% of actual churners. Among all thresholds that satisfy this recall requirement, the threshold with the highest precision will be selected.

This approach aims to catch most churners while reducing unnecessary false-positive alerts.

The **80% recall requirement is a project assumption**, not a known business requirement. In a real deployment, this value should be determined together with stakeholders based on factors such as:

* the cost of losing a customer,
* the cost of a retention intervention,
* the capacity of the retention team,
* and the acceptable number of false-positive alerts.

If these costs become available later, threshold selection could be improved further using an **expected-cost calculation**.

Threshold selection will be performed using **training/validation data rather than the held-out test set**. The test set will remain untouched until both the final model and its operating threshold have been selected.
