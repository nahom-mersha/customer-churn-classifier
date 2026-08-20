import numpy as np

from customer_churn_classifier.logistic_numpy import train_logistic_regression

X = np.array(
    [
        [0.0],
        [1.0],
        [2.0],
        [3.0],
    ]
)

y = np.array([0, 0, 1, 1])


for learning_rate in [0.001, 0.1, 10.0]:
    _, _, losses = train_logistic_regression(
        X,
        y,
        learning_rate=learning_rate,
        n_iterations=100,
    )

    print(f"Learning rate: {learning_rate}")
    print(f"First loss: {losses[0]}")
    print(f"Final loss: {losses[-1]}")
    print()

for learning_rate in [0.001, 0.1, 10.0]:
    _, _, losses = train_logistic_regression(
        X,
        y,
        learning_rate=learning_rate,
        n_iterations=100,
    )

    print(f"Learning rate: {learning_rate}")
    print("First 10 losses:", losses[:10])
    print("Final loss:", losses[-1])
    print()
