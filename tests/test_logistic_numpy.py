import numpy as np

from customer_churn_classifier.logistic_numpy import (
    compute_gradients,
    linear_score,
    numerical_bias_gradient,
    numerical_weight_gradients,
    sigmoid,
)


def test_analytical_gradients_match_numerical_gradients() -> None:
    X = np.array(
        [
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 4.0],
        ]
    )

    y = np.array([0.0, 0.0, 1.0])

    w = np.array([0.1, -0.2])
    b = 0.05

    z = linear_score(X, w, b)
    p = sigmoid(z)

    dw_analytical, db_analytical = compute_gradients(X, y, p)

    dw_numerical = numerical_weight_gradients(X, y, w, b)
    db_numerical = numerical_bias_gradient(X, y, w, b)

    assert np.allclose(
        dw_analytical,
        dw_numerical,
        rtol=1e-5,
        atol=1e-7,
    )

    assert np.isclose(
        db_analytical,
        db_numerical,
        rtol=1e-5,
        atol=1e-7,
    )
