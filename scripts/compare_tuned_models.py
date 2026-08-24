from sklearn.model_selection import StratifiedKFold, cross_validate

from customer_churn_classifier.data import load_clean_data
from customer_churn_classifier.models import (
    build_gradient_boosting_pipeline,
    build_logistic_pipeline,
)
from customer_churn_classifier.split import make_train_test_split


def main() -> None:
    df = load_clean_data()

    X_train, _X_test, y_train, _y_test = make_train_test_split(df)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    logistic = build_logistic_pipeline()
    logistic.set_params(
        classifier__C=1.0,
    )

    gradient_boosting = build_gradient_boosting_pipeline()
    gradient_boosting.set_params(
        classifier__learning_rate=0.1,
        classifier__max_depth=1,
        classifier__n_estimators=200,
    )

    models = {
        "logistic_regression": logistic,
        "gradient_boosting": gradient_boosting,
    }

    scoring = [
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]

    for name, pipeline in models.items():
        results = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
        )

        print(f"\n{name}")

        for metric in scoring:
            scores = results[f"test_{metric}"]

            print(f"{metric}: {scores.mean():.4f} ± {scores.std():.4f}")


if __name__ == "__main__":
    main()
