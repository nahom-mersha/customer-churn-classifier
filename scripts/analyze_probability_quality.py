from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
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
        reports_dir = Path("reports/figures")
    reports_dir.mkdir(parents=True, exist_ok=True)

    model_probabilities = {
        "Logistic Regression": logistic_proba,
        "Gradient Boosting": gradient_boosting_proba,
    }

    # ROC curve
    fig, ax = plt.subplots()

    for model_name, probabilities in model_probabilities.items():
        RocCurveDisplay.from_predictions(
            y_train,
            probabilities,
            name=model_name,
            ax=ax,
        )

    ax.set_title("ROC Curve: Logistic Regression vs Gradient Boosting")
    fig.tight_layout()
    fig.savefig(reports_dir / "roc_curve.png", dpi=300)

    # Precision-Recall curve
    fig, ax = plt.subplots()

    for model_name, probabilities in model_probabilities.items():
        PrecisionRecallDisplay.from_predictions(
            y_train,
            probabilities,
            name=model_name,
            ax=ax,
        )

    ax.set_title("Precision-Recall Curve: Logistic Regression vs Gradient Boosting")
    fig.tight_layout()
    fig.savefig(reports_dir / "precision_recall_curve.png", dpi=300)
    # Calibration curve / reliability diagram
    fig, ax = plt.subplots()

    for model_name, probabilities in model_probabilities.items():
        CalibrationDisplay.from_predictions(
            y_train,
            probabilities,
            n_bins=10,
            strategy="quantile",
            name=model_name,
            ax=ax,
        )

    ax.set_title("Calibration Curve: Logistic Regression vs Gradient Boosting")
    fig.tight_layout()
    fig.savefig(reports_dir / "calibration_curve.png", dpi=300)

    threshold = 0.50
    gradient_boosting_pred = (gradient_boosting_proba >= threshold).astype(int)

    analysis_df = X_train.copy()
    analysis_df["y_true"] = y_train.to_numpy()
    analysis_df["churn_probability"] = gradient_boosting_proba
    analysis_df["y_pred"] = gradient_boosting_pred

    analysis_df["error_type"] = "correct"
    analysis_df.loc[
        (analysis_df["y_true"] == 0) & (analysis_df["y_pred"] == 1),
        "error_type",
    ] = "false_positive"
    analysis_df.loc[
        (analysis_df["y_true"] == 1) & (analysis_df["y_pred"] == 0),
        "error_type",
    ] = "false_negative"

    print()
    print("Error counts at threshold 0.50:")
    print(analysis_df["error_type"].value_counts())

    print()
    print("Mean numeric features by error type:")
    print(
        analysis_df.groupby("error_type")[
            ["tenure", "MonthlyCharges", "TotalCharges"]
        ].mean()
    )

    print()
    print("Contract distribution by error type:")
    print(
        analysis_df.groupby("error_type")["Contract"]
        .value_counts(normalize=True)
        .rename("share")
    )

    print()
    print("InternetService distribution by error type:")
    print(
        analysis_df.groupby("error_type")["InternetService"]
        .value_counts(normalize=True)
        .rename("share")
    )

    print()
    print("PaymentMethod distribution by error type:")
    print(
        analysis_df.groupby("error_type")["PaymentMethod"]
        .value_counts(normalize=True)
        .rename("share")
    )


if __name__ == "__main__":
    main()
