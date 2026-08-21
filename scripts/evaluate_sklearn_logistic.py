from customer_churn_classifier.data import load_clean_data
from customer_churn_classifier.models import build_logistic_pipeline
from customer_churn_classifier.split import make_train_test_split


def main() -> None:
    df = load_clean_data()

    X_train, X_test, y_train, y_test = make_train_test_split(df)

    pipeline = build_logistic_pipeline()

    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    print("Training complete")
    print(f"Number of test samples: {len(X_test)}")
    print(f"Probability shape: {probabilities.shape}")
    print(f"Minimum probability: {probabilities.min():.4f}")
    print(f"Maximum probability: {probabilities.max():.4f}")
    print("First 5 churn probabilities:")
    print(probabilities[:5])


if __name__ == "__main__":
    main()
