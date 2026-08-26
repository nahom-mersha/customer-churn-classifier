# Customer Churn Classifier Roadmap

**Status: Completed.**
This document records the project scope,
implementation plan, and completion criteria. See the README for
final results and usage instructions.

This project builds a customer churn classification system. Given customer contract, billing, and service information, the system estimates the probability that a customer will churn.

The project starts with problem definition and dataset validation, then moves through preprocessing, baselines, from-scratch logistic regression, professional model comparison, probability evaluation, threshold selection, and deployment-ready inference.

## Project Goals

- Define churn as a supervised binary classification problem.
- Use a public telecom churn dataset responsibly.
- Build a reproducible data-cleaning and preprocessing workflow.
- Implement logistic regression from scratch with NumPy.
- Compare the NumPy model with professional scikit-learn classifiers.
- Evaluate models beyond accuracy using precision, recall, F1, ROC-AUC, PR-AUC, calibration, and business-cost-aware thresholds.
- Save a final model pipeline for consistent inference.
- Provide batch prediction and API-style prediction workflows.
- Document limitations, intended use, and non-use cases.

## 1. Define the Classification Problem

Clarify the real-world problem before modeling.

Decisions:

- One row represents one customer.
- The target is whether the customer churned.
- The model predicts churn probability, not only a hard yes/no label.
- A retention or customer-success team could use high-risk predictions to prioritize review or outreach.
- False negatives miss customers who leave.
- False positives spend effort on customers who may have stayed anyway.
- Accuracy alone is not enough because churn is a minority class.

Done when the prediction unit, target, user, action, errors, and success criteria are clear.

## 2. Choose and Document the Dataset

Use the classic IBM Telco Customer Churn dataset.

Dataset summary:

- 7,043 fictional telecom customer records.
- Target column: `Churn`.
- Features include tenure, contract type, billing, payment method, monthly charges, total charges, and subscribed services.
- The dataset is useful for learning binary classification, categorical preprocessing, class imbalance, and probability evaluation.
- The dataset is fictional and does not provide a perfect temporal prediction window.

Done when the dataset source, target, feature meaning, class balance, limitations, and leakage risks are documented.

## 3. Set Up Reproducible Data Structure

Create a clean project structure from the reusable AI project template.

Key structure:

- `src/customer_churn_classifier/` for reusable project code.
- `scripts/` for runnable project commands.
- `data/raw/` for downloaded raw data.
- `data/processed/` for generated cleaned data.
- `reports/` for generated analysis reports.
- `configs/` for project settings.
- `tests/` for automated tests.

Raw and processed data files are not committed to GitHub.

Done when the project can load the selected dataset through a documented command.

## 4. Explore, Clean, and Validate the Data

Inspect the dataset and generate a data-quality report.

Checks:

- shape and column types;
- target distribution;
- missing values;
- duplicate rows;
- categorical values;
- numeric ranges;
- validation checks for target values, tenure, and charges.

Cleaning decisions:

- Convert blank `TotalCharges` values to missing values.
- Convert `TotalCharges` to numeric.
- Encode `Churn` as `No = 0` and `Yes = 1`.
- Keep `customerID` for reference but exclude it from model features.

Done when `reports/data_quality_report.md` explains the key data-quality findings and cleaning decisions.

## 5. Design Preprocessing Without Data Leakage

Build preprocessing that is fitted only on training data.

Preprocessing decisions:

- Numeric features are imputed with the training-set median.
- Numeric features are scaled.
- Categorical missing values are filled with `missing`.
- Categorical features are one-hot encoded.
- Unknown future categories are ignored safely.
- `customerID` is excluded from model features.
- `Churn` is separated as the target.

Leakage rule:

The train/test split happens before fitting imputers, scalers, encoders, calibration methods, or thresholds. The test set is transformed using preprocessing values learned only from the training set.

Done when the split and preprocessing pipeline can run without using test-set information.

## 6. Build the Majority-Class Baseline

Create the simplest possible classifier: always predict the most common training class.

Evaluate:

- accuracy;
- confusion matrix;
- churn-class precision;
- churn-class recall;
- F1 score.

Purpose:

This baseline shows why accuracy can be misleading. If most customers do not churn, a model can get high accuracy while failing to identify churners.

Done when baseline results are saved and explained.

## 7. Implement Logistic Regression From Scratch

Implement logistic regression with NumPy.

Core pieces:

- linear score: `z = Xw + b`;
- sigmoid function;
- binary cross-entropy loss;
- vectorized gradients;
- gradient descent training;
- probability prediction;
- threshold-based label prediction.

Done when the NumPy model trains on controlled data, produces probabilities between 0 and 1, and shows decreasing loss.

## 8. Verify Gradients and Optimization

Use numerical gradient checking to verify the analytical gradients.

Experiments:

- compare analytical gradients with finite-difference gradients;
- test learning rates that are too small, suitable, and too large;
- compare behavior with scaled and unscaled features.

Done when gradient differences are small and optimization behavior can be explained.

## 9. Evaluate the From-Scratch Model

Evaluate the NumPy logistic regression model on the held-out test set.

Checks:

- confusion matrix;
- precision, recall, and F1;
- threshold experiments such as `0.30`, `0.50`, and `0.70`;
- saved predicted probabilities.

Done when the effect of different thresholds on false positives and false negatives is clear.

## 10. Build Professional Scikit-Learn Pipelines

Train scikit-learn models using the same leakage-safe preprocessing logic.

Start with professional logistic regression through a pipeline.

Done when a reproducible scikit-learn logistic regression workflow runs through one clean evaluation path.

## 11. Compare Professional Models

Compare several classifiers fairly.

Models:

- majority baseline;
- scikit-learn logistic regression;
- K-nearest neighbors;
- decision tree;
- random forest;
- gradient boosting.

Compare using the same split, preprocessing logic, cross-validation strategy, and metrics.

Done when a comparison table explains which model should move forward and why.

## 12. Analyze Probability Quality and Errors

Evaluate whether predicted probabilities are useful, not just labels.

Analysis:

- ROC-AUC;
- precision-recall curve;
- PR-AUC;
- calibration curve;
- Brier score;
- false positives and false negatives;
- segment-level error patterns;
- feature effects or permutation importance.

Done when ranking quality, probability quality, and failure modes are documented.

## 13. Select a Business-Cost-Aware Threshold

Choose a decision threshold using explicit business assumptions.

Example assumptions:

- cost of contacting a customer;
- value of retaining a true churner;
- cost of missing a churner.

Compare thresholds by expected cost or expected value.

Done when the final threshold is justified by stated assumptions instead of using `0.50` by default.

## 14. Train and Save the Final Model Artifact

Train the selected pipeline and save it with metadata.

The saved artifact should include:

- preprocessing;
- classifier;
- feature list;
- random seed;
- selected threshold;
- important metrics;
- training date or version.

Done when a clean checkout can train and save the model artifact with one documented command.

## 15. Ship Batch Prediction and API Prediction

Create usable prediction workflows.

Batch prediction:

- read a CSV of customer records;
- validate required columns;
- output churn probabilities and labels.

API-style prediction:

- accept one customer record;
- return churn probability, threshold, and predicted label;
- return clear errors for invalid inputs.

Done when a user can run documented prediction commands.

## 16. Add Engineering Quality and Documentation

Add tests, logging, README documentation, and a model card.

Tests should cover:

- data cleaning;
- preprocessing;
- split behavior;
- NumPy logistic regression;
- gradient checking;
- training and prediction behavior;
- invalid inputs.

Documentation should include setup, data source, commands, model results, limitations, intended use, and non-use cases.

Done when tests pass and the project is reproducible from the README.

## 17. Final Verification and Portfolio Review

Run the full workflow end to end.

Verify:

- raw data to cleaned data;
- data-quality report;
- preprocessing;
- baselines;
- NumPy logistic regression;
- model comparison;
- calibration and threshold choice;
- saved model;
- batch/API prediction;
- tests and documentation.

Done when the repository is reproducible, documented, and ready to explain in interviews or portfolio reviews.

## Completion Definition

The project is complete when it contains a reproducible customer churn classification workflow with:

- documented problem definition and dataset choice;
- cleaned and validated data;
- leakage-safe preprocessing;
- majority-class baseline;
- NumPy logistic regression with gradient checking;
- professional model comparison;
- probability-focused evaluation;
- calibration and business-cost threshold selection;
- saved model artifact;
- batch and API prediction workflows;
- tests, logging, README, and model card.