from __future__ import division
from __future__ import print_function

from scipy.stats import norm

import numpy as np

from . import common_args
from ..util import read_param_file


def analyze(problem, X, Y, num_resamples=1000,
            conf_level=0.95, print_to_console=False):
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

    D = problem['num_vars']

    if Y.size % (D + 1) == 0:
        N = int(Y.size / (D + 1))
    else:
        raise RuntimeError("Incorrect number of samples in model output file.")

    if not 0 < conf_level < 1:
        raise RuntimeError("Confidence level must be between 0-1.")

    base = np.zeros(N)
    X_base = np.zeros((N, D))
    perturbed = np.zeros((N, D))
    X_perturbed = np.zeros((N, D))
    step = D + 1

    base = Y[0:Y.size:step]
    X_base = X[0:Y.size:step, :]
    for j in range(D):
        perturbed[:, j] = Y[(j + 1):Y.size:step]
        X_perturbed[:, j] = X[(j + 1):Y.size:step, j]

    # First order (+conf.) and Total order (+conf.)
    keys = ('vi', 'vi_std', 'dgsm', 'dgsm_conf')
    S = dict((k, np.zeros(D)) for k in keys)
    if print_to_console:
        print("Parameter %s %s %s %s" % keys)

    for j in range(D):
        S['vi'][j], S['vi_std'][j] = calc_vi(
            base, perturbed[:, j], X_perturbed[:, j] - X_base[:, j])
        S['dgsm'][j], S['dgsm_conf'][j] = calc_dgsm(base, perturbed[:, j], X_perturbed[
                                                    :, j] - X_base[:, j], problem['bounds'][j], num_resamples, conf_level)

        if print_to_console:
            print("%s %f %f %f %f" % (
                problem['names'][j], S['vi'][j], S['vi_std'][j], S['dgsm'][j], S['dgsm_conf'][j]))

    return S


def calc_vi(base, perturbed, x_delta):
    # v_i sensitivity measure following Sobol and Kucherenko (2009)
    # For comparison, Morris mu* < sqrt(v_i)
    dfdx = (perturbed - base) / x_delta
    dfdx2 = dfdx ** 2

    return np.mean(dfdx2), np.std(dfdx2)


def calc_dgsm(base, perturbed, x_delta, bounds, num_resamples, conf_level):
    # v_i sensitivity measure following Sobol and Kucherenko (2009)
    # For comparison, total order S_tot <= dgsm
    D = np.var(base)
    vi, _ = calc_vi(base, perturbed, x_delta)
    dgsm = vi * (bounds[1] - bounds[0]) ** 2 / (D * np.pi ** 2)

    s = np.zeros(num_resamples)
    for i in range(num_resamples):
        r = np.random.randint(len(base), size=len(base))
        s[i], _ = calc_vi(base[r], perturbed[r], x_delta[r])

    return dgsm, norm.ppf(0.5 + conf_level / 2) * s.std(ddof=1)

if __name__ == "__main__":
    parser = common_args.create()
    parser.add_argument('-X', '--model-input-file', type=str,
                        required=True, default=None, help='Model input file')
    parser.add_argument('-r', '--resamples', type=int, required=False, default=1000,
                        help='Number of bootstrap resamples for Sobol confidence intervals')
    args = parser.parse_args()
    problem = read_param_file(args.paramfile)

    Y = np.loadtxt(args.model_output_file, delimiter=args.delimiter, usecols=(args.column,))
    X = np.loadtxt(args.model_input_file, delimiter=args.delimiter, ndmin=2)
    if len(X.shape) == 1:
        X = X.reshape((len(X), 1))

    analyze(problem, X, Y, num_resamples=args.resamples, print_to_console=True)
