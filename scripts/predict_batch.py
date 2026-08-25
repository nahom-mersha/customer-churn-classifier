import argparse
from pathlib import Path

from customer_churn_classifier.batch_prediction import (
    DEFAULT_CONFIG_PATH,
    predict_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate batch customer churn predictions."
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to input CSV containing customer feature columns.",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path where prediction output CSV will be written.",
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

    predict_batch(
        input_path=args.input,
        output_path=args.output,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
