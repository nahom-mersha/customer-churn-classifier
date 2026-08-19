import numpy as np


def linear_score(
    X: np.ndarray,
    w: np.ndarray,
    b: float,
) -> np.ndarray:
    """Compute the linear score Xw + b."""
    return X @ w + b


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Compute the sigmoid function in a numerically stable way."""
    result = np.empty_like(z, dtype=float)

    positive = z >= 0
    negative = ~positive

    result[positive] = 1 / (1 + np.exp(-z[positive]))

    exp_z = np.exp(z[negative])
    result[negative] = exp_z / (1 + exp_z)

    return result
