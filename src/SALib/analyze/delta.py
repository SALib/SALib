from typing import Dict
from scipy.stats import norm, gaussian_kde, rankdata

import numpy as np

from . import common_args
from ..util import read_param_file, ResultDict


def analyze(
    problem: Dict,
    X: np.ndarray,
    Y: np.ndarray,
    num_resamples: int = 100,
    conf_level: float = 0.95,
    print_to_console: bool = False,
    seed: int = None,
    y_resamples: int = None,
    method: str = "all",
) -> Dict:
    """Perform Delta Moment-Independent Analysis on model outputs.

    Returns a dictionary with keys 'delta', 'delta_conf', 'S1', and 'S1_conf'
    (first-order sobol indices), where each entry is a list of size D
    (the number of parameters) containing the indices in the same order as the
    parameter file.


    Notes
    -----
    Compatible with:
        all samplers


    Examples
    --------
        >>> X = latin.sample(problem, 1000)
        >>> Y = Ishigami.evaluate(X)
        >>> Si = delta.analyze(problem, X, Y, print_to_console=True)


    Parameters
    ----------
    problem : dict
        The problem definition
    X: numpy.matrix
        A NumPy matrix containing the model inputs
    Y : numpy.array
        A NumPy array containing the model outputs
    num_resamples : int
        The number of resamples when computing confidence intervals (default 100)
    conf_level : float
        The confidence interval level (default 0.95)
    print_to_console : bool
        Print results directly to console (default False)
    y_resamples : int, optional
        Number of samples to use when resampling (bootstrap) (default None)
    method : {"all", "delta", "sobol"}, optional
        Whether to compute "delta", "sobol" or both ("all") indices (default "all")


    References
    ----------
    1. Borgonovo, E. (2007). "A new uncertainty importance measure."
           Reliability Engineering & System Safety, 92(6):771-784,
           doi:10.1016/j.ress.2006.04.015.

    2. Plischke, E., E. Borgonovo, and C. L. Smith (2013). "Global
           sensitivity measures from given data." European Journal of
           Operational Research, 226(3):536-550, doi:10.1016/j.ejor.2012.11.047.
    """
    if seed:
        np.random.seed(seed)

    D = problem["num_vars"]
    if y_resamples is None:
        y_resamples = Y.size

    if not y_resamples <= Y.size:
        raise ValueError(
            "y_resamples must be less than or equal to the total number of samples"
        )

    if not 0 < conf_level < 1:
        raise RuntimeError("Confidence level must be between 0-1.")

    # equal frequency partition
    exp = 2.0 / (7.0 + np.tanh((1500.0 - y_resamples) / 500.0))
    M = int(np.round(min(int(np.ceil(y_resamples**exp)), 48)))
    m = np.linspace(0, y_resamples, M + 1)
    Ygrid = np.linspace(np.min(Y), np.max(Y), 100)

    keys = ("delta", "delta_conf", "S1", "S1_conf")
    S = ResultDict((k, np.zeros(D)) for k in keys)
    S["names"] = problem["names"]

    try:
        for i in range(D):
            X_i = X[:, i]
            if method in ["all", "delta"]:
                S["delta"][i], S["delta_conf"][i] = bias_reduced_delta(
                    Y, Ygrid, X_i, m, num_resamples, conf_level, y_resamples
                )
            if method in ["all", "sobol"]:
                ind = np.random.randint(Y.size, size=y_resamples)
                S["S1"][i] = sobol_first(Y[ind], X_i[ind], m)
                S["S1_conf"][i] = sobol_first_conf(
                    Y, X_i, m, num_resamples, conf_level, y_resamples
                )
    except np.linalg.LinAlgError as e:
        msg = "Singular matrix detected\n"
        msg += "This may be due to the sample size ({}) being too small\n".format(
            Y.size
        )
        msg += "If this is not the case, check Y values or raise an issue with the\n"
        msg += "SALib team"

        raise np.linalg.LinAlgError(msg) from e

    if print_to_console:
        print(S.to_df())

    return S


def calc_delta(Y, Ygrid, X, m):
    """Plischke et al. (2013) delta index estimator (eqn 26) for d_hat."""
    N = len(Y)
    fy = gaussian_kde(Y, bw_method="silverman")(Ygrid)
    xr = rankdata(X, method="ordinal")

    d_hat = 0.0
    l_m = len(m) - 1
    for j in range(l_m):
        ix = np.where((xr > m[j]) & (xr <= m[j + 1]))[0]
        nm = len(ix)

        # if not np.all(np.equal(Y_ix, Y_ix[0])):
        Y_ix = Y[ix]
        if Y_ix.ptp() != 0.0:
            fyc = gaussian_kde(Y_ix, bw_method="silverman")(Ygrid)
            fy_ = np.abs(fy - fyc)
        else:
            fy_ = np.abs(fy)

        d_hat += (nm / (2 * N)) * np.trapz(fy_, Ygrid)

    return d_hat


def bias_reduced_delta(Y, Ygrid, X, m, num_resamples, conf_level, y_resamples):
    """Plischke et al. 2013 bias reduction technique (eqn 30)"""
    d = np.empty(num_resamples)

    N = len(Y)
    ind = np.random.randint(N, size=y_resamples)
    d_hat = calc_delta(Y[ind], Ygrid, X[ind], m)
    r = np.random.randint(N, size=(num_resamples, y_resamples))

    for i in range(num_resamples):
        r_i = r[i, :]
        d[i] = calc_delta(Y[r_i], Ygrid, X[r_i], m)

    d = 2.0 * d_hat - d
    return (d.mean(), norm.ppf(0.5 + conf_level / 2) * d.std(ddof=1))


def sobol_first(Y, X, m):
    # pre-process to catch constant array
    # see: https://github.com/numpy/numpy/issues/9631
    if Y.ptp() == 0.0:
        # Catch constant results
        # If Y does not change then it is not sensitive to anything...
        return 0.0

    xr = rankdata(X, method="ordinal")
    Vi = 0
    N = len(Y)
    Y_mean = Y.mean()
    for j in range(len(m) - 1):
        ix = np.where((xr > m[j]) & (xr <= m[j + 1]))[0]
        nm = len(ix)
        Vi += (nm / N) * ((Y[ix].mean() - Y_mean) ** 2)

    return Vi / np.var(Y)


def sobol_first_conf(Y, X, m, num_resamples, conf_level, y_resamples):
    s = np.zeros(num_resamples)

    N = len(Y)
    r = np.random.randint(N, size=(num_resamples, y_resamples))

    for i in range(num_resamples):
        r_i = r[i, :]
        s[i] = sobol_first(Y[r_i], X[r_i], m)

    return norm.ppf(0.5 + conf_level / 2) * s.std(ddof=1)


def cli_parse(parser):
    parser.add_argument(
        "-X",
        "--model-input-file",
        type=str,
        required=True,
        default=None,
        help="Model input file",
    )
    parser.add_argument(
        "-r",
        "--resamples",
        type=int,
        required=False,
        default=10,
        help="Number of bootstrap resamples for \
                           Sobol confidence intervals",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        required=False,
        default="all",
        help="Method to compute sensitivities \
                    'delta', 'sobol' or 'all'",
    )
    parser.add_argument(
        "--y_resamples",
        type=int,
        required=False,
        default=None,
        help="Number of samples to use when \
                    resampling (bootstrap)",
    )
    return parser


def cli_action(args):
    problem = read_param_file(args.paramfile)
    Y = np.loadtxt(
        args.model_output_file, delimiter=args.delimiter, usecols=(args.column,)
    )
    X = np.loadtxt(args.model_input_file, delimiter=args.delimiter, ndmin=2)
    if len(X.shape) == 1:
        X = X.reshape((len(X), 1))

    analyze(
        problem,
        X,
        Y,
        num_resamples=args.resamples,
        print_to_console=True,
        seed=args.seed,
        method=args.method,
        y_resamples=args.y_resamples,
    )


if __name__ == "__main__":
    common_args.run_cli(cli_parse, cli_action)
