import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from customer_churn_classifier.data import load_clean_data
from customer_churn_classifier.models import (
    build_gradient_boosting_pipeline,
    build_logistic_pipeline,
)
from customer_churn_classifier.split import make_train_test_split
from customer_churn_classifier.thresholds import evaluate_thresholds


def main() -> None:
    df = load_clean_data()

    X_train, X_test, y_train, y_test = make_train_test_split(df)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    logistic = build_logistic_pipeline()
    logistic.set_params(classifier__C=1.0)

    gradient_boosting = build_gradient_boosting_pipeline()
    gradient_boosting.set_params(
        classifier__learning_rate=0.1,
        classifier__max_depth=1,
        classifier__n_estimators=200,
    )

    thresholds = np.arange(0.10, 0.91, 0.05)

    models = {
        "logistic_regression": logistic,
        "gradient_boosting": gradient_boosting,
    }

    for model_name, model in models.items():
        probabilities = cross_val_predict(
            model,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
        )[:, 1]

        threshold_results = evaluate_thresholds(
            y_true=y_train.to_numpy(),
            probabilities=probabilities,
            thresholds=thresholds,
        )

        best_row = threshold_results.loc[threshold_results["net_value"].idxmax()]

        print()
        print(model_name)
        print(threshold_results)
        print()
        print("Best threshold:")
        print(best_row)


if __name__ == "__main__":
    main()
