from sklearn.model_selection import StratifiedKFold, cross_validate

from customer_churn_classifier.data import load_clean_data
from customer_churn_classifier.models import build_logistic_pipeline
from customer_churn_classifier.split import make_train_test_split


def main() -> None:
    df = load_clean_data()

    X_train, X_test, y_train, y_test = make_train_test_split(df)

    pipeline = build_logistic_pipeline()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=[
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "average_precision",
        ],
    )
    # final held-out test evaluation comes later
    for metric in [
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]:
        scores = results[f"test_{metric}"]
        print(f"{metric}: {scores.mean():.4f}")


if __name__ == "__main__":
    main()
