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

    X : np.array
        An array containing the model inputs of dtype=float

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
    p = problem['num_vars']

    assert X.shape[0] == Y.shape[0], \
        "X and Y must be of corresponding size (number of X values must match number of Y values)"

    assert (X.shape[0] / sample_sets) == p+1, \
        "Number of parameter set groups must match number of parameters + 1"

    assert (Y.shape[0] / sample_sets) == p+1, \
        "Number of result set groups must match number of parameters + 1"

    if seed:
        np.random.set_seed(seed)

    st = np.empty((p, sample_sets))

    # Each `n`th item from 0-position is the baseline for
    # that N group.
    nth = p+1
    Y_base = Y[0::nth]
    r = sample_sets
    for i in range(p):
        pos = i + 1

        # Collect change for every `n`th element
        st[i] = Y_base - Y[pos::nth]
    # End for

    Si = ResultDict((k, [None] * p)
                    for k in ['names', 'ST', 'ST_conf'])


    Si['ST'] = ((1.0/(2.0*r)) * np.sum(st**2, axis=1))
    Si['ST_conf'] = compute_radial_si_confidence(st, r, num_resamples,
                                                 conf_level)
    Si['names'] = problem['names']

    return Si


def compute_radial_si_confidence(si: np.array, N: int, num_resamples: int,
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
    res = np.var(si_resampled, axis=0) - ((1.0/(2.0*N)) * np.sum(si_resampled**2, axis=0))

    return norm.ppf(0.5 + conf_level / 2.0) * res.std(ddof=1, axis=0)
