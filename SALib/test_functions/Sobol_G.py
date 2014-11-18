from __future__ import division
import numpy as np

# Non-monotonic Sobol G Function (8 parameters)
# First-order indices:
# x1: 0.7165
# x2: 0.1791
# x3: 0.0237
# x4: 0.0072
# x5-x8: 0.0001


def evaluate(values,a=[0, 1, 4.5, 9, 99, 99, 99, 99]):

    if type(values) != np.ndarray:
        raise TypeError("The argument `values` must be a numpy ndarray")

    ltz = np.array(values) < 0
    gtz = np.array(values) > 0

    if ltz.any() == True:
        raise ValueError("Sobol G function called with values less than one")
    elif gtz.any() == True:
        raise ValueError("Sobol G function called with values greater than one")

    Y = np.empty([values.shape[0]])

    for i, row in enumerate(values):
        Y[i] = 1.0

        for j in range(len(a)):
            x = row[j]
            Y[i] *= (abs(4 * x - 2) + a[j]) / (1 + a[j])

    return Y
