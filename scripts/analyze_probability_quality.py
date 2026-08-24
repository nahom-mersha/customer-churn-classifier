from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from customer_churn_classifier.data import load_clean_data
from customer_churn_classifier.models import (
    build_gradient_boosting_pipeline,
    build_logistic_pipeline,
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

    logistic = build_logistic_pipeline()
    logistic.set_params(classifier__C=1.0)

    gradient_boosting = build_gradient_boosting_pipeline()
    gradient_boosting.set_params(
        classifier__learning_rate=0.1,
        classifier__max_depth=1,
        classifier__n_estimators=200,
    )

    logistic_proba = cross_val_predict(
        logistic,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    gradient_boosting_proba = cross_val_predict(
        gradient_boosting,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
    )[:, 1]

    for model_name, probabilities in {
        "logistic_regression": logistic_proba,
        "gradient_boosting": gradient_boosting_proba,
    }.items():
        roc_auc = roc_auc_score(y_train, probabilities)
        average_precision = average_precision_score(y_train, probabilities)
        brier = brier_score_loss(y_train, probabilities)

        print()
        print(model_name)
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"Average Precision: {average_precision:.4f}")
        print(f"Brier score: {brier:.4f}")


if __name__ == "__main__":
    main()
