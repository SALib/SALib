import numpy as np
from scipy.stats import norm
from typing import Dict, Optional

from SALib.sample import common_args
from ..util import read_param_file, ResultDict

__all__ = ['analyze']

def analyze(problem: Dict, Y: np.array, 
            sample_sets: int,
            num_resamples: int = 1000,
            conf_level: float = 0.95,
            seed: Optional[int] = None) -> ResultDict:
    """Estimation of Total Sensitivity Index using the Jansen Sensitivity Estimator.

    Compatible with:
    * `radial_sobol`
    * `radial_mc`

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
    Si : ResultDict


    Usage Example
    -------------
    ```python
    >>> from SALib.sample.radial.radial_sobol import sample
    >>> X = sample(problem, N, seed)
    >>> r_Y = Sobol_G.evaluate(X)
    >>> sobol_jansen.analyze(problem, Y, N, num_resamples=1000)
    ```

    References
    ----------
    .. [1] Campolongo, F., Saltelli, A., Cariboni, J., 2011. 
           From screening to quantitative sensitivity analysis: A unified 
           approach. Computer Physics Communications 182, 978–988.
           https://www.sciencedirect.com/science/article/pii/S0010465510005321
           DOI: 10.1016/j.cpc.2010.12.039

    .. [2] M.J.W. Jansen, Analysis of variance designs for model output, 
           Computer Physics Communication 117 (1999) 35–43.
           https://www.sciencedirect.com/science/article/pii/S0010465598001544
           DOI: 10.1016/S0010-4655(98)00154-4

    .. [3] M.J.W. Jansen, W.A.H. Rossing, R.A. Daamen, Monte Carlo estimation
           of uncertainty contributions from several independent multivariate
           sources, in: J. Gasmanand, G. van Straten (Eds.), Predictability
           and Nonlinear Modelling in Natural Sciences and Economics,
           Kluwer Academic Publishers, Dordrecht, 1994, pp. 334–343
           DOI: 10.1007/978-94-011-0962-8_28
    """
    num_vars = problem['num_vars']

    assert (Y.shape[0] / sample_sets) == num_vars + 1, \
        "Number of result set groups must match number of parameters + 1"

    if seed:
        np.random.seed(seed)

    st = np.zeros((sample_sets, num_vars))

    # Each `n`th item from 0-position is the baseline for that N group.
    nth = num_vars + 1
    Y_base = Y[0::nth]

    base_variance = np.var(Y_base)
    for i in range(num_vars):
        pos = i + 1

        # Collect change for every `n`th element
        st[:, i] = Y_base - Y[pos::nth]


    Si = ResultDict((k, [None] * num_vars)
                    for k in ['names', 'ST', 'ST_conf'])

    Si['ST'] = (jansen_estimator(sample_sets, st) / base_variance)
    Si['ST_conf'] = compute_radial_si_confidence(st, sample_sets,
                                                 base_variance,
                                                 num_resamples,
                                                 conf_level)
    Si['names'] = problem['names']

    return Si


def compute_radial_si_confidence(si: np.array, N: int,
                                 base_variance: np.array,
                                 num_resamples: int = 1000,
                                 conf_level: float = 0.95) -> np.array:
    '''Uses bootstrapping where the sensitivity are resampled with
    replacement to produce a histogram of resampled metrics.
    This resample is used to produce a confidence interval.

    Largely identical to `morris.compute_mu_star_confidence`.
    Modified calculate conf for all parameters in one step.

    Arguments
    ---------
    si : np.array
        The sensitivity effect for each parameter

    N : int
        The number of sample sets used to create `si`

    num_resamples : int
        The number of bootstrap resamples of `ST` (default 1000)

    conf_level : float
        The confidence interval level (default 0.95)

    Returns
    ---------
    conf : np.array
        Confidence bounds for mu_star for each parameter
    '''
    if not 0 < conf_level < 1:
        raise ValueError("Confidence level must be between 0 and 1.")

    if len(si.shape) > 1:
        num_params = si.shape[1]
    else:
        num_params = 1

    idx = np.random.randint(len(si), size=(num_resamples, num_params))

    resampled = jansen_estimator(N, si[idx]) / base_variance
    return norm.ppf(0.5 + conf_level / 2.0) * resampled.std(ddof=1, axis=0)


def jansen_estimator(N: int, si: np.array) -> np.array:
    """
    Jansen sensitivity estimator.

    Arguments
    ---------
    N : int
        Number of sample sets.

    si : np.array
        Values of (Y_base - Y_perturbed)
    
    Returns
    -------
    jansen estimates : np.array
    """
    return (1.0/(2.0*N)) * np.sum((si**2), axis=0)


def cli_parse(parser):
    parser.add_argument('-n', '--sample_sets',
                        type=int, required=True, help='Number of sample sets used')
    parser.add_argument('-r', '--num_resamples', type=int, required=False,
                        default=1000,
                        help='Number of bootstrap resamples for confidence intervals')
    parser.add_argument('-L', '--conf_level', type=float, required=False,
                        default=0.95,
                        help='The confidence interval level (default: 0.95)')
    return parser


def cli_action(args):
    problem = read_param_file(args.paramfile)
    Y = np.loadtxt(args.model_output_file,
                   delimiter=args.delimiter,
                   usecols=(args.column,))

    analyze(problem, Y, args.sample_sets,
            conf_level=args.conf_level,
            num_resamples=args.num_resamples,
            seed=args.seed)


if __name__ == "__main__":
    common_args.run_cli(cli_parse, cli_action)
