from sklearn.model_selection import StratifiedKFold, cross_validate

from customer_churn_classifier.data import load_clean_data
from customer_churn_classifier.models import (
    build_decision_tree_pipeline,
    build_gradient_boosting_pipeline,
    build_knn_pipeline,
    build_logistic_pipeline,
    build_random_forest_pipeline,
)
from customer_churn_classifier.split import make_train_test_split


def main() -> None:
    df = load_clean_data()

    X_train, X_test, y_train, y_test = make_train_test_split(df)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    models = {
        "logistic_regression": build_logistic_pipeline(),
        "knn": build_knn_pipeline(),
        "decision_tree": build_decision_tree_pipeline(),
        "random_forest": build_random_forest_pipeline(),
        "gradient_boosting": build_gradient_boosting_pipeline(),
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

            mean_score = scores.mean()
            std_score = scores.std()

            print(f"{metric}: {mean_score:.4f} ± {std_score:.4f}")


if __name__ == "__main__":
    main()
