from __future__ import division
import numpy as np

# Non-monotonic Sobol G Function (8 parameters)
# First-order indices:
# x1: 0.7165
# x2: 0.1791
# x3: 0.0237
# x4: 0.0072
# x5-x8: 0.0001


def evaluate(values):
    a = [0, 1, 4.5, 9, 99, 99, 99, 99]
    Y = np.empty([values.shape[0]])

    for i, row in enumerate(values):
        Y[i] = 1.0

        for j in range(8):
            x = row[j]
            Y[i] *= (abs(4 * x - 2) + a[j]) / (1 + a[j])

    return Y
