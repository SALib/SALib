import numpy as np


def evaluate(values):
    """Linear model (#1) used in Li et al., (2010).

        y = x1 + x2 + x3 + x4 + x5

    Parameters
    ----------
    values : np.array


    References
    ----------
    .. [1] Genyuan Li, H. Rabitz, P.E. Yelvington, O.O. Oluwole, F. Bacon,
            C.E. Kolb, and J. Schoendorf, "Global Sensitivity Analysis for
            Systems with Independent and/or Correlated Inputs", Journal of
            Physical Chemistry A, Vol. 114 (19), pp. 6022 - 6032, 2010,
            https://doi.org/10.1021/jp9096919
    """

    Y = np.zeros([values.shape[0]])
    Y = np.sum(values,axis=1)

    return Y
