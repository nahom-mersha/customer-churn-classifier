import argparse
import tempfile
from pathlib import Path

import pandas as pd

from customer_churn_classifier.batch_prediction import (
    DEFAULT_CONFIG_PATH,
    predict_batch,
)
from customer_churn_classifier.snowflake_data import load_curated_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load curated Snowflake data and generate churn predictions."
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path where cloud prediction CSV will be saved.",
    )

    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        type=Path,
        help="Path to final model config file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading curated data from Snowflake...")
    cloud_data = load_curated_data()

    print(f"Rows loaded from Snowflake: {len(cloud_data)}")

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_directory = Path(temporary_directory)

        input_path = temporary_directory / "snowflake_curated_input.csv"
        prediction_path = temporary_directory / "predictions.csv"

        # Save the Snowflake DataFrame temporarily because the existing
        # batch prediction function expects a CSV input file.
        cloud_data.to_csv(input_path, index=False)

        predict_batch(
            input_path=input_path,
            output_path=prediction_path,
            config_path=args.config,
        )

        predictions = pd.read_csv(prediction_path)

    # Rename the original target column so it is clear that this is the
    # known historical outcome, not a model prediction.
    predictions = predictions.rename(columns={"Churn": "actual_churn"})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output, index=False)

    print("Cloud pipeline completed successfully.")
    print(f"Output file: {args.output}")
    print(f"Rows predicted: {len(predictions)}")
    print("Output columns include:")
    print("customerID")
    print("churn_probability")
    print("predicted_churn")
    print("decision_threshold")
    print("actual_churn")


if __name__ == "__main__":
    main()
