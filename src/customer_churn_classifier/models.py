from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from customer_churn_classifier.preprocessing import build_preprocessor


def build_logistic_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("classifier", LogisticRegression()),
        ]
    )
