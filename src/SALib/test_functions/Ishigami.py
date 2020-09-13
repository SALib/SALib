import math

import numpy as np


# Non-monotonic Ishigami Function (3 parameters)
# Using Saltelli sampling with a sample size of ~1000
# the expected first-order indices would be:
# x1: 0.3139
# x2: 0.4424
# x3: 0.0
def evaluate(values):
    Y = np.zeros(values.shape[0])
    A = 7
    B = 0.1

    # X = values
    # Y = np.sin(X[:, 0]) + A * np.power(np.sin(X[:, 1]), 2) + \
    #         B * np.power(X[:, 2], 4) * np.sin(X[:, 0])
    for i, X in enumerate(values):
        Y[i] = np.sin(X[0]) + A * np.power(np.sin(X[1]), 2) + \
            B * np.power(X[2], 4) * np.sin(X[0])

    return Y
