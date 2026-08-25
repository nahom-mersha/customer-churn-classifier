import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import yaml
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

CONFIG_PATH = Path("configs/final_model.yaml")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load final model training configuration."""
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main() -> None:
    config = load_config()

    threshold = config["model"]["threshold"]
    hyperparameters = config["model"]["hyperparameters"]

    model_path = Path(config["artifacts"]["model_path"])
    metadata_path = Path(config["artifacts"]["metadata_path"])
    overwrite = config["artifacts"]["overwrite"]

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    if not overwrite and model_path.exists():
        raise FileExistsError(f"Model artifact already exists: {model_path}")

    if not overwrite and metadata_path.exists():
        raise FileExistsError(f"Metadata artifact already exists: {metadata_path}")

    df = load_clean_data()
    X_train, X_test, y_train, y_test = make_train_test_split(df)

    model = build_gradient_boosting_pipeline()
    model.set_params(
        classifier__learning_rate=hyperparameters["learning_rate"],
        classifier__max_depth=hyperparameters["max_depth"],
        classifier__n_estimators=hyperparameters["n_estimators"],
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()
    net_value = int(180 * tp - 20 * fp - 180 * fn)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "average_precision": average_precision_score(y_test, probabilities),
        "brier_score": brier_score_loss(y_test, probabilities),
        "net_value": net_value,
    }

    metadata = {
        "model_name": config["model"]["name"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "threshold": threshold,
        "hyperparameters": hyperparameters,
        # Recorded for reproducibility metadata.
        # The current selected model builder sets random_state=42 internally.
        "random_seed": config["data"]["random_seed"],
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "features": list(X_train.columns),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "metrics": metrics,
        "artifact_paths": {
            "model_path": str(model_path),
            "metadata_path": str(metadata_path),
        },
    }

    joblib.dump(model, model_path)

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("Final model trained and saved.")
    print(f"Model artifact: {model_path}")
    print(f"Metadata artifact: {metadata_path}")
    print()
    print("Final test metrics:")
    for metric_name, metric_value in metrics.items():
        if metric_name == "net_value":
            print(f"{metric_name}: €{metric_value:,}")
        else:
            print(f"{metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
