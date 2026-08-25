from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from customer_churn_classifier.data import load_clean_data
from customer_churn_classifier.models import build_gradient_boosting_pipeline
from customer_churn_classifier.split import make_train_test_split

SELECTED_THRESHOLD = 0.10


def main() -> None:
    df = load_clean_data()

    X_train, X_test, y_train, y_test = make_train_test_split(df)

    model = build_gradient_boosting_pipeline()
    model.set_params(
        classifier__learning_rate=0.1,
        classifier__max_depth=1,
        classifier__n_estimators=200,
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= SELECTED_THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    net_value = 180 * tp - 20 * fp - 180 * fn

    print("Final held-out test evaluation")
    print()
    print("Selected model: Gradient Boosting")
    print(f"Selected threshold: {SELECTED_THRESHOLD:.2f}")
    print()
    print("Confusion matrix counts:")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}")
    print()
    print("Classification metrics:")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, predictions):.4f}")
    print(f"F1: {f1_score(y_test, predictions):.4f}")
    print()
    print("Probability/ranking metrics:")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")
    print(f"Average Precision: {average_precision_score(y_test, probabilities):.4f}")
    print(f"Brier score: {brier_score_loss(y_test, probabilities):.4f}")
    print()
    print("Business metric:")
    print(f"Net value: €{net_value:,}")
    print()
    print(
        "Note: this is a final test evaluation. It is not used to retune the threshold."
    )


if __name__ == "__main__":
    main()
