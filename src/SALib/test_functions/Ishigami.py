from __future__ import division

import math

import numpy as np


# Non-monotonic Ishigami Function (3 parameters)
# First-order indices:
# x1: 0.3139
# x2: 0.4424
# x3: 0.0
def evaluate(values):
    Y = np.zeros([values.shape[0]])
    A = 7
    B = 0.1

    for i, X in enumerate(values):
        Y[i] = math.sin(X[0]) + A * math.pow(math.sin(X[1]), 2) + \
            B * math.pow(X[2], 4) * math.sin(X[0])

    return Y
