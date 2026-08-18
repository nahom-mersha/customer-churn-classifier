from sklearn.model_selection import train_test_split

from customer_churn_classifier.data import TARGET_COLUMN
from customer_churn_classifier.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
)

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def split_features_target(df):
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def make_train_test_split(
    df,
    test_size=0.2,
    random_state=42,
):
    X, y = split_features_target(df)

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
