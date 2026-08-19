from customer_churn_classifier.baseline import run_majority_baseline


def main() -> None:
    results = run_majority_baseline()

    print("Majority-class baseline results")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1: {results['f1']:.4f}")
    print(f"Confusion matrix: {results['confusion_matrix']}")


if __name__ == "__main__":
    main()
