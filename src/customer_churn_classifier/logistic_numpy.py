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


def binary_cross_entropy(
    y_true: np.ndarray,
    p: np.ndarray,
) -> float:
    """Compute average binary cross-entropy loss."""
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)

    losses = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

    return float(np.mean(losses))


def compute_gradients(
    X: np.ndarray,
    y: np.ndarray,
    p: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Compute gradients for logistic regression weights and bias."""
    m = X.shape[0]
    error = p - y

    dw = (X.T @ error) / m
    db = float(np.mean(error))

    return dw, db
