from __future__ import division
from __future__ import print_function

from scipy.stats import norm, gaussian_kde, rankdata

import numpy as np

from . import common_args
from ..util import read_param_file


def analyze(problem, X, Y, num_resamples=10,
            conf_level=0.95, print_to_console=False):
    """Perform Delta Moment-Independent Analysis on model outputs.
    
    Returns a dictionary with keys 'delta', 'delta_conf', 'S1', and 'S1_conf',
    where each entry is a list of size D (the number of parameters) containing
    the indices in the same order as the parameter file.
    
    Parameters
    ----------
    problem : dict
        The problem definition
    X: numpy.matrix
        A NumPy matrix containing the model inputs
    Y : numpy.array
        A NumPy array containing the model outputs
    num_resamples : int
        The number of resamples when computing confidence intervals (default 10)
    conf_level : float
        The confidence interval level (default 0.95)
    print_to_console : bool
        Print results directly to console (default False)
        
    References
    ----------
    .. [1] Borgonovo, E. (2007). "A new uncertainty importance measure."
           Reliability Engineering & System Safety, 92(6):771-784,
           doi:10.1016/j.ress.2006.04.015.
           
    .. [2] Plischke, E., E. Borgonovo, and C. L. Smith (2013). "Global
           sensitivity measures from given data." European Journal of
           Operational Research, 226(3):536-550, doi:10.1016/j.ejor.2012.11.047.
           
    Examples
    --------
    >>> X = latin.sample(problem, 1000)
    >>> Y = Ishigami.evaluate(X)
    >>> Si = delta.analyze(problem, X, Y, print_to_console=True)
    """

    D = problem['num_vars']
    N = Y.size

    if not 0 < conf_level < 1:
        raise RuntimeError("Confidence level must be between 0-1.")

    # equal frequency partition
    M = min(np.ceil(N ** (2 / (7 + np.tanh((1500 - N) / 500)))), 48)
    m = np.linspace(0, N, M + 1)
    Ygrid = np.linspace(np.min(Y), np.max(Y), 100)

    keys = ('delta', 'delta_conf', 'S1', 'S1_conf')
    S = dict((k, np.zeros(D)) for k in keys)
    if print_to_console:
        print("Parameter %s %s %s %s" % keys)

    for i in range(D):
        S['delta'][i], S['delta_conf'][i] = bias_reduced_delta(
            Y, Ygrid, X[:, i], m, num_resamples, conf_level)
        S['S1'][i] = sobol_first(Y, X[:, i], m)
        S['S1_conf'][i] = sobol_first_conf(
            Y, X[:, i], m, num_resamples, conf_level)
        if print_to_console:
            print("%s %f %f %f %f" % (problem['names'][i], S['delta'][
                  i], S['delta_conf'][i], S['S1'][i], S['S1_conf'][i]))

    return S

# Plischke et al. 2013 estimator (eqn 26) for d_hat

def calc_delta(Y, Ygrid, X, m):
    N = len(Y)
    fy = gaussian_kde(Y, bw_method='silverman')(Ygrid)
    xr = rankdata(X, method='ordinal')

    d_hat = 0
    for j in range(len(m) - 1):
        ix = np.where((xr > m[j]) & (xr <= m[j + 1]))[0]
        nm = len(ix)
        fyc = gaussian_kde(Y[ix], bw_method='silverman')(Ygrid)
        d_hat += (nm / (2 * N)) * np.trapz(np.abs(fy - fyc), Ygrid)

    return d_hat

# Plischke et al. 2013 bias reduction technique (eqn 30)

def bias_reduced_delta(Y, Ygrid, X, m, num_resamples, conf_level):
    d = np.zeros(num_resamples)
    d_hat = calc_delta(Y, Ygrid, X, m)

    for i in range(num_resamples):
        r = np.random.randint(len(Y), size=len(Y))
        d[i] = calc_delta(Y[r], Ygrid, X[r], m)

    d = 2 * d_hat - d
    return (d.mean(), norm.ppf(0.5 + conf_level / 2) * d.std(ddof=1))


def sobol_first(Y, X, m):
    xr = rankdata(X, method='ordinal')
    Vi = 0
    N = len(Y)
    for j in range(len(m) - 1):
        ix = np.where((xr > m[j]) & (xr <= m[j + 1]))[0]
        nm = len(ix)
        Vi += (nm / N) * (Y[ix].mean() - Y.mean()) ** 2
    return Vi / np.var(Y)


def sobol_first_conf(Y, X, m, num_resamples, conf_level):
    s = np.zeros(num_resamples)

    for i in range(num_resamples):
        r = np.random.randint(len(Y), size=len(Y))
        s[i] = sobol_first(Y[r], X[r], m)

    return norm.ppf(0.5 + conf_level / 2) * s.std(ddof=1)

if __name__ == "__main__":
    parser = common_args.create()
    parser.add_argument('-X', '--model-input-file', type=str,
                        required=True, default=None, help='Model input file')
    parser.add_argument('-r', '--resamples', type=int, required=False, default=10,
                        help='Number of bootstrap resamples for Sobol confidence intervals')
    args = parser.parse_args()

    problem = read_param_file(args.paramfile)
    Y = np.loadtxt(args.model_output_file, delimiter=args.delimiter, usecols=(args.column,))
    X = np.loadtxt(args.model_input_file, delimiter=args.delimiter, ndmin=2)
    if len(X.shape) == 1:
        X = X.reshape((len(X), 1))

    analyze(problem, X, Y, num_resamples=args.resamples, print_to_console=True)
