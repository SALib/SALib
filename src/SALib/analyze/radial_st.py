import numpy as np
from scipy.stats import norm
from typing import Dict, Optional

from ..util import ResultDict

__all__ = ['analyze']

def analyze(problem: Dict, Y: np.array, sample_sets: int,
            num_resamples: int = 1000,
            conf_level: float = 0.95,
            seed: Optional[int] = None) -> Dict:
    """Global Sensitivity Index for `radial` approach.

    Sample size must be of a compatible number, or else results
    will not be sane.

    Arguments
    ---------
    problem : dict
        The SALib problem specification

    Y : np.array
        An array containing the model outputs of dtype=float

    sample_sets : int
        The number of sample sets used to create `X`

    num_resamples : int
        The number of resamples to calculate `mu_star_conf` (default 1000)

    conf_level : float
        The confidence interval level (default 0.95)

    seed : int
        Seed value to use for np.random.seed

    Returns
    --------
    Si : dict
    """
    num_vars = problem['num_vars']

    assert (Y.shape[0] / sample_sets) == num_vars + 1, \
        "Number of result set groups must match number of parameters + 1"

    if seed:
        np.random.set_seed(seed)

    st = np.empty((num_vars, sample_sets))

    # Each `n`th item from 0-position is the baseline for
    # that N group.
    nth = num_vars + 1
    Y_base = Y[0::nth]

    base_variance = np.var(Y_base)
    r = sample_sets
    for i in range(num_vars):
        pos = i + 1

        # Collect change for every `n`th element
        st[i] = Y_base - Y[pos::nth]
    # End for

    Si = ResultDict((k, [None] * num_vars)
                    for k in ['names', 'ST', 'ST_conf'])

    jansen_estim = jansen_estimator(r, st.T)
    Si['ST'] = (jansen_estim / base_variance)
    Si['ST_conf'] = compute_radial_si_confidence(st, base_variance, r, num_resamples,
                                                 conf_level)
    Si['names'] = problem['names']

    return Si


def compute_radial_si_confidence(si: np.array, base_var: np.array, 
                                 N: int, 
                                 num_resamples: int,
                                 conf_level: float = 0.95) -> np.array:
    '''Uses bootstrapping where the elementary effects are resampled with
    replacement to produce a histogram of resampled mu_star metrics.
    This resample is used to produce a confidence interval.

    Largely identical to `morris.compute_mu_star_confidence`.
    Modified calculate conf for all parameters in one go

    Arguments
    ---------
    si : np.array
        The sensitivity effect for each parameter

    N : int
        The number of sample sets used

    num_resamples : int
        The number of resamples to calculate `mu_star_conf` (default 1000)

    conf_level : float
        The confidence interval level (default 0.95)

    Returns
    ---------
    conf : np.array
        Confidence bounds for mu_star for each parameter
    '''
    tmp_si = si.T
    if not 0 < conf_level < 1:
        raise ValueError("Confidence level must be between 0-1.")

    resample_index = np.random.randint(tmp_si.shape[0], size=(num_resamples, N))

    si_resampled = tmp_si[resample_index]
    res = (jansen_estimator(N, si_resampled) / base_var)

    return norm.ppf(0.5 + conf_level / 2.0) * res.std(ddof=1, axis=0)


def jansen_estimator(N: int, si: np.array):
    """
    Arguments
    ---------
    N : int
        Number of sample sets (repetitions)
    si : np.array
        1D array
    """
    return (1.0/(2.0*N)) * np.sum((si**2), axis=0)
