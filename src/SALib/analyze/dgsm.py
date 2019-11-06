from __future__ import division
from __future__ import print_function

from scipy.stats import norm

import numpy as np

from . import common_args
from ..util import read_param_file, ResultDict


def analyze(problem, X, Y, num_resamples=1000,
            conf_level=0.95, print_to_console=False, seed=None):
    """Calculates Derivative-based Global Sensitivity Measure on model outputs.

    Returns a dictionary with keys 'vi', 'vi_std', 'dgsm', and 'dgsm_conf',
    where each entry is a list of size D (the number of parameters) containing
    the indices in the same order as the parameter file.

    Parameters
    ----------
    problem : dict
        The problem definition
    X : numpy.matrix
        The NumPy matrix containing the model inputs
    Y : numpy.array
        The NumPy array containing the model outputs
    num_resamples : int
        The number of resamples used to compute the confidence
        intervals (default 1000)
    conf_level : float
        The confidence interval level (default 0.95)
    print_to_console : bool
        Print results directly to console (default False)

    References
    ----------
    .. [1] Sobol, I. M. and S. Kucherenko (2009). "Derivative based global
           sensitivity measures and their link with global sensitivity
           indices." Mathematics and Computers in Simulation, 79(10):3009-3017,
           doi:10.1016/j.matcom.2009.01.023.
    """
    if seed:
        np.random.seed(seed)

    D = problem['num_vars']
    Y_size = Y.size

    if Y_size % (D + 1) == 0:
        N = int(Y_size / (D + 1))
    else:
        raise RuntimeError("Incorrect number of samples in model output file.")

    if not 0 < conf_level < 1:
        raise RuntimeError("Confidence level must be between 0-1.")

    dims = (N, D)
    base = np.empty(N)
    X_base = np.empty(dims)
    perturbed = np.empty(dims)
    X_perturbed = np.empty(dims)
    step = D + 1

    base = Y[0:Y_size:step]
    X_base = X[0:Y_size:step, :]

    # First order (+conf.) and Total order (+conf.)
    keys = ('vi', 'vi_std', 'dgsm', 'dgsm_conf')
    S = ResultDict((k, np.empty(D)) for k in keys)
    S['names'] = problem['names']

    if print_to_console:
        print("Parameter %s %s %s %s" % keys)

    bounds = problem['bounds']
    for j in range(D):
        perturbed[:, j] = Y[(j + 1):Y_size:step]
        X_perturbed[:, j] = X[(j + 1):Y_size:step, j]

        diff = X_perturbed[:, j] - X_base[:, j]
        perturbed_j = perturbed[:, j]
        S['vi'][j], S['vi_std'][j] = calc_vi_stats(base,
                                                   perturbed_j,
                                                   diff)
        S['dgsm'][j], S['dgsm_conf'][j] = calc_dgsm(base,
                                                    perturbed_j,
                                                    diff,
                                                    bounds[j],
                                                    num_resamples,
                                                    conf_level)

        if print_to_console:
            print("%s %f %f %f %f" % (
                S['names'][j], S['vi'][j], S['vi_std'][j], S['dgsm'][j], S['dgsm_conf'][j]))

    return S


def calc_vi_stats(base, perturbed, x_delta):
    """Calculate v_i mean and std.
    
    v_i sensitivity measure following Sobol and Kucherenko (2009)
    For comparison, Morris mu* < sqrt(v_i)

    Same as calc_vi_mean but returns standard deviation as well.
    """
    dfdx = ((perturbed - base) / x_delta)**2
    return np.mean(dfdx), np.std(dfdx)


def calc_vi_mean(base, perturbed, x_delta):
    """Calculate v_i mean.

    Same as calc_vi_stats but only returns the mean.
    """
    dfdx = ((perturbed - base) / x_delta)**2
    return dfdx.mean()


def calc_dgsm(base, perturbed, x_delta, bounds, num_resamples, conf_level):
    """v_i sensitivity measure following Sobol and Kucherenko (2009).
    For comparison, total order S_tot <= dgsm
    """
    D = np.var(base)
    vi = calc_vi_mean(base, perturbed, x_delta)
    dgsm = vi * (bounds[1] - bounds[0])**2 / (D * np.pi**2)

    len_base = len(base)
    s = np.empty(num_resamples)
    r = np.random.randint(len_base, size=(num_resamples, len_base))
    for i in range(num_resamples):
        r_i = r[i]
        s[i] = calc_vi_mean(base[r_i], perturbed[r_i], x_delta[r_i])

    return dgsm, norm.ppf(0.5 + conf_level / 2.0) * s.std(ddof=1)


def cli_parse(parser):
    parser.add_argument('-X', '--model-input-file', type=str,
                        required=True, default=None,
                        help='Model input file')
    parser.add_argument('-r', '--resamples', type=int, required=False,
                        default=1000,
                        help='Number of bootstrap resamples for Sobol \
                           confidence intervals')
    return parser


def cli_action(args):
    problem = read_param_file(args.paramfile)

    Y = np.loadtxt(args.model_output_file,
                   delimiter=args.delimiter, usecols=(args.column,))
    X = np.loadtxt(args.model_input_file, delimiter=args.delimiter, ndmin=2)
    if len(X.shape) == 1:
        X = X.reshape((len(X), 1))

    analyze(problem, X, Y, num_resamples=args.resamples, print_to_console=True,
            seed=args.seed)


if __name__ == "__main__":
    common_args.run_cli(cli_parse, cli_action)
