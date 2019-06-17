from __future__ import division

import numpy as np


# Non-monotonic Sobol G Function (8 parameters)
# First-order indices:
# x1: 0.7165
# x2: 0.1791
# x3: 0.0237
# x4: 0.0072
# x5-x8: 0.0001
def evaluate(values, a=None):
    if type(values) != np.ndarray:
        raise TypeError("The argument `values` must be a numpy ndarray")
    if a is None:
        a = [0, 1, 4.5, 9, 99, 99, 99, 99]

    ltz = values < 0
    gto = values > 1

    if ltz.any() == True:
        raise ValueError("Sobol G function called with values less than zero")
    elif gto.any() == True:
        raise ValueError("Sobol G function called with values greater than one")

    Y = np.ones([values.shape[0]])

    len_a = len(a)
    for i, row in enumerate(values):
        for j in range(len_a):
            x = row[j]
            a_j = a[j]
            Y[i] *= (np.abs(4 * x - 2) + a_j) / (1 + a_j)

    return Y


def partial_first_order_variance(a=None):
    if a is None:
        a = [0, 1, 4.5, 9, 99, 99, 99, 99]
    a = np.array(a)
    return np.divide(1, np.multiply(3, np.square(1 + a)))


def total_variance(a=None):
    if a is None:
        a = [0, 1, 4.5, 9, 99, 99, 99, 99]
    a = np.array(a)
    return np.add(-1, np.product(1 + partial_first_order_variance(a), axis=0))


def sensitivity_index(a):
    a = np.array(a)
    return np.divide(partial_first_order_variance(a), total_variance(a))


def total_sensitivity_index(a):
    a = np.array(a)
    
    pv = partial_first_order_variance(a)
    tv = total_variance(a)
    
    sum_pv = pv.sum(axis=0)
    
    return np.subtract(1, np.divide(np.subtract(sum_pv, pv.T), tv))
