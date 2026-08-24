from sklearn.model_selection import GridSearchCV, StratifiedKFold

from customer_churn_classifier.data import load_clean_data
from customer_churn_classifier.models import build_logistic_pipeline
from customer_churn_classifier.split import make_train_test_split


def main() -> None:
    df = load_clean_data()

    X_train, _X_test, y_train, _y_test = make_train_test_split(df)

    pipeline = build_logistic_pipeline()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    param_grid = {
        "classifier__C": [
            0.01,
            0.1,
            1.0,
            10.0,
        ],
    }

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="average_precision",
        cv=cv,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    print("Best parameters:")
    print(search.best_params_)

    print(f"Best mean CV average precision: {search.best_score_:.4f}")


if __name__ == "__main__":
    main()
