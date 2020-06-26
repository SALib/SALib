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
    values : numpy.ndarray
        input variables
    a : numpy.ndarray
        parameter values
    delta : numpy.ndarray
        shift parameters
    alpha : numpy.ndarray
        curvature parameters

    Returns
    -------
    Y : Result of G-function
    """
    if type(values) != np.ndarray:
        raise TypeError("The argument `values` must be a numpy ndarray")
  
    if a is None:
        a = np.array([0, 1, 4.5, 9, 99, 99, 99, 99])

    if delta is None:
        delta = np.zeros_like(a)
    else:
        if not isinstance(delta, np.ndarray):
            raise TypeError("The argument `delta` must be given as a numpy ndarray")

        delta_inbetween = delta[(delta < 0) | (delta > 1)]
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

    for i, row in enumerate(values):
        shift_of_x = row + delta
        integral = np.modf(shift_of_x)[1]
        mod_x = shift_of_x - integral
        temp_y = (np.abs(2 * mod_x - 1)**alpha)
        y_elements = ((1 + alpha) * temp_y + a) / (1 + a)
        Y[i] = np.prod(y_elements)

    return Y



def _partial_first_order_variance(a=None, alpha=None):
    if a is None:
        a = [0, 1, 4.5, 9, 99, 99, 99, 99]
    if alpha is None:
        alpha = np.ones_like(a)
    a = np.array(a)
    
    return np.divide((alpha**2), np.multiply((1 + 2 * alpha), np.square(1 + a)))


def _total_variance(a=None, alpha=None):
    if a is None:
        a = [0, 1, 4.5, 9, 99, 99, 99, 99]
    if alpha is None:
        alpha = np.ones_like(a)

    a = np.array(a)
    return np.add(-1, np.product(1 + _partial_first_order_variance(a, alpha), axis=0))


def sensitivity_index(a, alpha=None):
    a = np.array(a)
    return np.divide(_partial_first_order_variance(a, alpha), _total_variance(a, alpha))


def total_sensitivity_index(a, alpha=None):
    a = np.array(a)
    
    pv = _partial_first_order_variance(a, alpha)
    tv = _total_variance(a, alpha)
    
    sum_pv = pv.sum(axis=0)
    
    return np.subtract(1, np.divide(np.subtract(sum_pv, pv.T), tv))


def V_Ti_regular(V: np.array, i: int):
    result = 1.0
    for j in range(len(V)):
        if j != i:
            result = result * (1.0 + V[j])
    return result * V[i]


def _calc_analytic(a: np.array, alpha: np.array, num_params: int):
    """Calculate analytic values for modified Sobol_G function
    """
    V_total = _total_variance(a, alpha)
    V_partial = _partial_first_order_variance(a, alpha)

    return np.array([np.round(V_Ti_regular(V_partial, i) / V_total, decimals=4)
                     for i in range(num_params)])
