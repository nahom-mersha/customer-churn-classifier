import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

DEFAULT_CONFIG_PATH = Path("configs/final_model.yaml")


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load final model configuration."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def load_metadata(path: Path) -> dict[str, Any]:
    """Load saved model metadata."""
    if not path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {path}. Run scripts/train_final_model.py first."
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_model(path: Path) -> Any:
    """Load saved preprocessing-and-model pipeline."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}. Run scripts/train_final_model.py first."
        )

    return joblib.load(path)


def validate_input_columns(
    input_data: pd.DataFrame,
    required_features: list[str],
) -> None:
    """Check that the input CSV contains all required feature columns."""
    missing_columns = [
        column for column in required_features if column not in input_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Input CSV is missing required columns: " + ", ".join(missing_columns)
        )


def predict_batch(
    input_path: Path,
    output_path: Path,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> None:
    """Load a saved model and create batch churn predictions."""
    config = load_config(config_path)

    model_path = Path(config["artifacts"]["model_path"])
    metadata_path = Path(config["artifacts"]["metadata_path"])

    metadata = load_metadata(metadata_path)
    model = load_model(model_path)

    required_features = metadata["features"]
    threshold = metadata["threshold"]

    input_data = pd.read_csv(input_path)
    validate_input_columns(input_data, required_features)

    features = input_data[required_features]

    probabilities = model.predict_proba(features)[:, 1]
    labels = (probabilities >= threshold).astype(int)

    output_data = input_data.copy()
    output_data["churn_probability"] = probabilities
    output_data["predicted_churn"] = labels
    output_data["decision_threshold"] = threshold

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data.to_csv(output_path, index=False)

    print("Batch predictions saved.")
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Model artifact: {model_path}")
    print(f"Metadata artifact: {metadata_path}")
    print(f"Rows predicted: {len(output_data)}")
    print(f"Decision threshold: {threshold}")
