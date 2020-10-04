import numpy as np


def evaluate(X: np.ndarray, A: float = 7.0, B: float = 0.1) -> np.ndarray:
    """Non-monotonic Ishigami-Homma three parameter test function:

    `f(x) = \sin(x_{1}) + A \sin(x_{2})^2 + Bx^{4}_{3}\sin(x_{1})`

    This test function is commonly used to benchmark global sensitivity 
    methods as variance-based sensitivities of this function can be 
    analytically determined.

    See listed references below.

    In [2], the expected first-order indices are:

        x1: 0.3139
        x2: 0.4424
        x3: 0.0

    when A = 7, B = 0.1 when conducting Sobol' analysis with the
    Saltelli sampling method with a sample size of 1000.


    Parameters
    ----------
    X : np.ndarray
        An `N*D` array holding values for each parameter, where `N` is the 
        number of samples and `D` is the number of parameters 
        (in this case, three).
    A : float
        Constant `A` parameter
    B : float
        Constant `B` parameter

    Returns
    -------
    Y : np.ndarray

    References
    ----------
    .. [1] Ishigami, T., Homma, T., 1990. 
           An importance quantification technique in uncertainty analysis for 
           computer models. 
           Proceedings. First International Symposium on Uncertainty Modeling 
           and Analysis. 
           https://doi.org/10.1109/ISUMA.1990.151285

    .. [2] Saltelli, A., Ratto, M., Andres, T., Campolongo, F., Cariboni, J., 
           Gatelli, D., Saisana, M., Tarantola, S., 2008. 
           Global Sensitivity Analysis: The Primer. Wiley, West Sussex, U.K.
           https://dx.doi.org/10.1002/9780470725184
    """
    Y = np.zeros(X.shape[0])
    Y = np.sin(X[:, 0]) + A * np.power(np.sin(X[:, 1]), 2) + \
            B * np.power(X[:, 2], 4) * np.sin(X[:, 0])

    return Y
