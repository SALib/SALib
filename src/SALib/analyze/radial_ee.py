from typing import Dict, Optional
import numpy as np
from scipy.stats import norm

from . import common_args
from ..util import read_param_file, ResultDict


__all__ = ['analyze']

def analyze(problem: Dict, X: np.array, Y: np.array, sample_sets: int,
            num_resamples: int = 100,
            conf_level: float = 0.95,
            print_to_console: bool = False,
            seed: Optional[int] = None) -> Dict:
    """Radial Elementary Effects Analysis.

    Calculates `mu`, `mu_star`, `sigma` and `mu_star_conf` as with
    Morris OAT using results from
    * `radial_sobol`
    * `radial_mc`

    Arguments
    ---------
    problem : dict
        The SALib problem specification
    
    X : np.array
        An array containing the model inputs of dtype=float

    Y : np.array
        An array containing the model outputs of dtype=float

    sample_sets : int
        The number of sample sets used to create result set `Y`

    num_resamples : int
        The number of resamples to calculate `mu_star_conf` (default 100)

    conf_level : float
        The confidence interval level (default 0.95)

    print_to_console : bool
        Print results directly to console (default False)

    seed : int
        Seed value to use for np.random.seed

    Returns
    --------
    Si : dict
    """
    num_vars = problem['num_vars']

    # Each `n`th item from 0-position is the baseline for
    # that N group.
    nth = num_vars + 1

    assert X.shape[0] == Y.shape[0], \
        "X and Y must be of corresponding size (number of X values must match number of Y values)"

    assert (X.shape[0] / sample_sets) == nth, \
        "Number of parameter set groups must match number of parameters + 1"

    assert (Y.shape[0] / sample_sets) == nth, \
        "Number of result set groups must match number of parameters + 1"

    if seed:
        np.random.seed(seed)

    ee = np.zeros((sample_sets, num_vars))
    
    X_base = X[0::nth]
    Y_base = Y[0::nth]
    for i in range(num_vars):
        pos = i + 1

        # Collect every `n`th element
        # which is the perturbation point
        x_tmp = (X_base[:, i] - X[pos::nth, i])

        try:
            ee[:, i] = (Y_base - Y[pos::nth]) / x_tmp
        except ZeroDivisionError:
            continue
    # End for

    Si = ResultDict((k, [None] * num_vars)
                    for k in ['names', 'mu', 'mu_star', 'sigma'])

    Si['mu'] = np.average(ee, axis=0)
    Si['mu_star'] = np.average(np.abs(ee), axis=0)

    Si['mu_star_conf'] = compute_radial_ee_confidence(ee, sample_sets, num_resamples, conf_level)

    Si['sigma'] = np.std(ee, ddof=1, axis=0)

    Si['names'] = problem['names']

    return Si


def compute_radial_ee_confidence(ee: np.array, N: int, num_resamples: int,
                                 conf_level: float = 0.95) -> np.array:
    '''Uses bootstrapping where the elementary effects are resampled with
    replacement to produce a histogram of resampled mu_star metrics.
    This resample is used to produce a confidence interval.

    Based on, and largely identical to, `morris.compute_mu_star_confidence`.

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
    if not 0 < conf_level < 1:
        raise ValueError("Confidence level must be between 0-1.")

    resample_index = np.random.randint(ee.shape[0], size=(num_resamples, N))
    ee_resampled = ee[resample_index]

    # Compute average of the absolute values over each of the resamples
    mu_star_resampled = np.average(np.abs(ee_resampled), axis=1)

    return norm.ppf(0.5 + conf_level / 2.0) * mu_star_resampled.std(ddof=1, axis=0)


def cli_parse(parser):
    parser.add_argument('-X', '--model-input-file', type=str, required=True,
                        default=None,
                        help='Model input file')
    parser.add_argument('-n', '--sample_sets',
                        type=int, required=True, help='Number of sample sets used')
    parser.add_argument('-r', '--resamples', type=int, required=False,
                        default=1000,
                        help='Number of bootstrap resamples for \
                              confidence intervals')
    parser.add_argument('-L', '--conf_level', type=float, required=False,
                        default=0.95,
                        help='The confidence interval level (default: 0.95)')
    return parser


def cli_action(args):
    problem = read_param_file(args.paramfile)

    X = np.loadtxt(args.model_input_file,
                   delimiter=args.delimiter)

    Y = np.loadtxt(args.model_output_file, delimiter=args.delimiter,
                   usecols=(args.column,))
    num_samples = args.sample_sets

    analyze(problem, X, Y, num_samples,
            num_resamples=args.num_resamples, 
            conf_level=args.conf_level,
            seed=args.seed)


if __name__ == '__main__':
    common_args.run_cli(cli_parse, cli_action)
