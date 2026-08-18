from customer_churn_classifier.data_report import (
    REPORT_PATH,
    load_processed_data,
    save_data_quality_report,
)


def main() -> None:
    df = load_processed_data()
    save_data_quality_report(df)

    print(f"Data quality report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
