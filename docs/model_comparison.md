# Professional Model Comparison

## Evaluation setup

The professional classifiers were compared using the same fixed training split and the same 5-fold stratified cross-validation procedure.

The held-out test set was not used during model comparison or hyperparameter tuning.

The evaluated models were:

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Gradient Boosting

The reported metrics were:

- Precision
- Recall
- F1
- ROC-AUC
- Average Precision

Average Precision was treated as the primary model-selection metric because churn is the minority class and precision-recall performance is especially relevant for this problem.

## Baseline cross-validation results

| Model | Precision | Recall | F1 | ROC-AUC | Average Precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.6529 ± 0.0255 | 0.5431 ± 0.0411 | 0.5923 ± 0.0296 | 0.8461 ± 0.0125 | 0.6614 ± 0.0195 |
| KNN | 0.5604 ± 0.0147 | 0.5351 ± 0.0212 | 0.5473 ± 0.0154 | 0.7827 ± 0.0066 | 0.5073 ± 0.0167 |
| Decision Tree | 0.4919 ± 0.0177 | 0.5057 ± 0.0211 | 0.4986 ± 0.0181 | 0.6583 ± 0.0124 | 0.3804 ± 0.0130 |
| Random Forest | 0.6215 ± 0.0351 | 0.4789 ± 0.0269 | 0.5408 ± 0.0281 | 0.8182 ± 0.0123 | 0.6107 ± 0.0296 |
| Gradient Boosting | 0.6606 ± 0.0289 | 0.5338 ± 0.0234 | 0.5902 ± 0.0227 | 0.8478 ± 0.0122 | 0.6671 ± 0.0232 |

The strongest baseline candidates were Logistic Regression and Gradient Boosting.

The single Decision Tree performed substantially worse than the ensemble-based tree models, while Random Forest improved considerably over the individual Decision Tree.

## Hyperparameter tuning

Hyperparameter tuning was performed using `GridSearchCV` on the training data only.

### Logistic Regression

The regularization parameter `C` was searched over:

```text
0.01
0.1
1.0
10.0
```

Best result:

```text
C = 1.0
Mean CV Average Precision = 0.6614
```

The default `C=1.0` remained the best value in the tested search space, so tuning did not improve Logistic Regression.

### Gradient Boosting

The following hyperparameters were searched:

```text
n_estimators  = [50, 100, 200]
learning_rate = [0.05, 0.1]
max_depth     = [1, 2, 3]
```

Best result:

```text
learning_rate = 0.1
max_depth = 1
n_estimators = 200
Mean CV Average Precision = 0.6705
```

This suggests that, for this dataset and search space, Gradient Boosting benefited from using many very simple weak trees.

## Tuned candidate comparison

| Metric | Logistic Regression | Gradient Boosting |
| --- | ---: | ---: |
| Precision | 0.6529 ± 0.0255 | 0.6737 ± 0.0340 |
| Recall | 0.5431 ± 0.0411 | 0.5278 ± 0.0262 |
| F1 | 0.5923 ± 0.0296 | 0.5914 ± 0.0239 |
| ROC-AUC | 0.8461 ± 0.0125 | 0.8505 ± 0.0125 |
| Average Precision | 0.6614 ± 0.0195 | 0.6705 ± 0.0208 |

Gradient Boosting achieved the highest Average Precision, ROC-AUC, and precision.

Logistic Regression achieved slightly higher recall, while the F1 scores were nearly identical.

The difference in Average Precision between the two models is relatively small compared with the cross-validation variability, so the result should not be interpreted as a decisive performance gap.

## Model-selection decision

Gradient Boosting will move forward as the primary candidate because it achieved the strongest cross-validated Average Precision, which is the primary ranking metric for this imbalanced churn problem.

Logistic Regression will remain as a reference candidate because:

- its overall performance is very close to Gradient Boosting;
- it achieved slightly higher recall;
- it is easier to interpret;
- it provides a useful comparison for probability calibration and feature-effect analysis.

The held-out test set has not been used to make this model-selection decision.