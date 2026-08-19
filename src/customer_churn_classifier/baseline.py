import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from customer_churn_classifier.data import PROCESSED_DATA_PATH
from customer_churn_classifier.split import make_train_test_split


def predict_majority_class(y_train: pd.Series, n_predictions: int) -> pd.Series:
    """Predict the most frequent training class for every row."""
    majority_class = y_train.mode()[0]

    return pd.Series([majority_class] * n_predictions)


def evaluate_majority_baseline(
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, object]:
    """Evaluate a majority-class baseline on the test labels."""
    y_pred = predict_majority_class(y_train, len(y_test))

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def run_majority_baseline(
    data_path=PROCESSED_DATA_PATH,
) -> dict[str, object]:
    """Load data, split it, and evaluate the majority-class baseline."""
    df = pd.read_csv(data_path)
    _, _, y_train, y_test = make_train_test_split(df)

    return evaluate_majority_baseline(y_train, y_test)
