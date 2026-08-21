import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from customer_churn_classifier.data import load_clean_data
from customer_churn_classifier.logistic_numpy import (
    predict_proba,
    train_logistic_regression,
)
from customer_churn_classifier.preprocessing import build_preprocessor
from customer_churn_classifier.split import make_train_test_split


def evaluate_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    """Evaluate binary predictions at a given probability threshold."""

    predictions = (probabilities >= threshold).astype(int)

    cm = confusion_matrix(y_true, predictions)
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)

    print(f"\nThreshold: {threshold:.2f}")
    print("Confusion matrix:")
    print(cm)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")


def main() -> None:
    # 1. Load cleaned churn data
    df = load_clean_data()

    # 2. Create the fixed stratified train/test split
    X_train, X_test, y_train, y_test = make_train_test_split(df)

    # 3. Fit preprocessing only on training data
    preprocessor = build_preprocessor()

    # Return dense arrays for the NumPy implementation
    preprocessor.set_params(categorical__one_hot_encoder__sparse_output=False)

    X_train_processed = np.asarray(preprocessor.fit_transform(X_train))

    X_test_processed = np.asarray(preprocessor.transform(X_test))

    # 4. Convert labels to NumPy arrays
    y_train_array = y_train.to_numpy(dtype=float)
    y_test_array = y_test.to_numpy(dtype=int)

    # 5. Train the from-scratch logistic regression model
    w, b, losses = train_logistic_regression(
        X_train_processed,
        y_train_array,
        learning_rate=0.1,
        n_iterations=1000,
    )

    # 6. Predict churn probabilities on the held-out test set
    probabilities = predict_proba(
        X_test_processed,
        w,
        b,
    )

    print("Training complete")
    print(f"First training loss: {losses[0]:.4f}")
    print(f"Final training loss: {losses[-1]:.4f}")

    # 7. Evaluate the same probabilities at different thresholds
    for threshold in [0.30, 0.50, 0.70]:
        evaluate_threshold(
            y_test_array,
            probabilities,
            threshold,
        )


if __name__ == "__main__":
    main()
