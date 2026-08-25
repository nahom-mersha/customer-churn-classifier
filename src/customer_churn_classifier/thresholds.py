import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def evaluate_thresholds(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    thresholds: np.ndarray,
) -> pd.DataFrame:
    """Evaluate classification thresholds using business net value.

    Business assumptions:
    - True positive: +180
    - False positive: -20
    - False negative: -180
    - True negative: 0
    """
    results = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()

        net_value = 180 * tp - 20 * fp - 180 * fn

        results.append(
            {
                "threshold": threshold,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "net_value": net_value,
            }
        )

    return pd.DataFrame(results)
