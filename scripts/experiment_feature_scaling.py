import numpy as np

from customer_churn_classifier.logistic_numpy import train_logistic_regression

# Same underlying pattern, but very different feature scales
X_unscaled = np.array(
    [
        [0.0, 0.0],
        [1.0, 100.0],
        [2.0, 200.0],
        [3.0, 300.0],
    ]
)

y = np.array([0, 0, 1, 1])


# Standardize each feature:
# x_scaled = (x - mean) / standard_deviation
mean = X_unscaled.mean(axis=0)
std = X_unscaled.std(axis=0)

X_scaled = (X_unscaled - mean) / std


learning_rate = 0.1
n_iterations = 100


_, _, losses_unscaled = train_logistic_regression(
    X_unscaled,
    y,
    learning_rate=learning_rate,
    n_iterations=n_iterations,
)

_, _, losses_scaled = train_logistic_regression(
    X_scaled,
    y,
    learning_rate=learning_rate,
    n_iterations=n_iterations,
)


print("Unscaled")
print("First 10 losses:", losses_unscaled[:10])
print("Final loss:", losses_unscaled[-1])

print()

print("Scaled")
print("First 10 losses:", losses_scaled[:10])
print("Final loss:", losses_scaled[-1])
