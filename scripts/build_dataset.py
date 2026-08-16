from customer_churn_classifier.data import build_processed_dataset


def main() -> None:
    cleaned = build_processed_dataset()

    print("Processed dataset created.")
    print(f"Rows: {cleaned.shape[0]}")
    print(f"Columns: {cleaned.shape[1]}")
    print(f"Churn distribution:\n{cleaned['Churn'].value_counts()}")
    print(f"Missing TotalCharges: {cleaned['TotalCharges'].isna().sum()}")


if __name__ == "__main__":
    main()
