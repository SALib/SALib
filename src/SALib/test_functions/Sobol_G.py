from __future__ import division

import numpy as np


# Non-monotonic Sobol G Function (8 parameters)
# First-order indices:
# x1: 0.7165
# x2: 0.1791
# x3: 0.0237
# x4: 0.0072
# x5-x8: 0.0001
def evaluate(values, a=None, delta=None, alpha=None):
    """Modified Sobol G-function.

    Reverts to original Sobol G-function if delta and alpha are not given.

    .. [1] Saltelli, A., Annoni, P., Azzini, I., Campolongo, F., Ratto, M., 
           Tarantola, S., 2010. Variance based sensitivity analysis of model 
           output. Design and estimator for the total sensitivity index. 
           Computer Physics Communications 181, 259–270. 
           https://doi.org/10.1016/j.cpc.2009.09.018

    Parameters
    ----------
    values : ndarray, input variables
    a : np.array,
    delta : np.array, of delta coefficients.
    alpha : np.array, of alpha coefficients

    Returns
    -------
    y : Scalar of G-function result.
    """
    if type(values) != np.ndarray:
        raise TypeError("The argument `values` must be a numpy ndarray")
    if delta and not isinstance(delta, np.ndarray):
        raise TypeError("The argument `delta` must be given as a numpy ndarray")
    if alpha and not isinstance(alpha, np.ndarray):
        raise TypeError("The argument `alpha` must be given as a numpy ndarray")

    if a is None:
        a = [0, 1, 4.5, 9, 99, 99, 99, 99]

    if delta is None:
        delta = np.zeros_like(a)
    else:
        if not isinstance(delta, np.ndarray):
            raise TypeError("The argument `delta` must be given as a numpy ndarray")

        delta_inbetween = (delta < 0) or (delta > 1)
        if delta_inbetween.any():
            raise ValueError("Sobol G function called with delta values less than zero or greater than one")

    if alpha is None:
        alpha = np.ones_like(a)
    else:
        if not isinstance(alpha, np.ndarray):
            raise TypeError("The argument `alpha` must be given as a numpy ndarray")

        alpha_gto = alpha <= 0.0
        if alpha_gto.any():
            raise ValueError("Sobol G function called with alpha values less than or equal to zero")


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
            alpha_j = alpha[j]

            shift_of_x = x + delta[j]
            integral = np.modf(shift_of_x)[1]
            mod_x = shift_of_x - integral
            
            temp_y = (np.abs(2 * mod_x - 1)**alpha_j)
            Y[i] *= ((1 + alpha_j) * temp_y + a_j)  / (1 + a_j)

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
