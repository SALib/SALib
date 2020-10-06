import numpy as np
from scipy.stats import norm, kstest

from . import common_args
from ..util import (read_param_file, ResultDict,
                    extract_group_names, _check_groups)


__all__ = ['analyze', 'cli_parse', 'cli_action']


def analyze(problem, X, Y, S=10, stat='median', 
            num_resamples: int = 100, conf_level: float = 0.95,
            print_to_console=False, seed=None):
    """Performs PAWN (2018) sensitivity analysis.

    PAWN is a density-based method which uses the cumulative distribution
    function (CDF), rather than the variance, of model outputs. Density-based
    methods may be more appropriate in cases where the variance of outputs
    does not well-represent uncertainty in the model's inputs. This may be 
    the case if the output distribution is highly-skewed or multi-modal [1].

    The PAWN method uses the two-sample Kolmogorov-Smirnov statistic to 
    measure the distance between unconditional and conditional CDFs (see [2]).

    This implementation was ported from [3], originally implemented in `R` 
    (https://github.com/baronig/GSA-cvd).


    Compatible with
    ---------------
    * all samplers


    Parameters
    ----------
    problem : dict
        The problem definition
    X : numpy.array
        A NumPy array containing the model inputs
    Y : numpy.array
        A NumPy array containing the model outputs
    S : int
        The number of equal-size intervals ('slides') to partition the 
        input-output space (default 10).
        The conditional mean (i.e. `E(Y|X_{i})`) is calculated for each 
        interval.
    stat : str or function
        If string, any numpy compatible statistic (defaults to `median`)
        A custom function can be provided instead.
    print_to_console : bool
        Print results directly to console (default False)
    seed : int
        Seed value to ensure deterministic results


    References
    ----------
    .. [1] Pianosi, F., Wagener, T., 2018. 
           Distribution-based sensitivity analysis from a generic input-output 
           sample. 
           Environmental Modelling & Software 108, 197–207. 
           https://doi.org/10.1016/j.envsoft.2018.07.019

    .. [2] Pianosi, F., Wagener, T., 2015. 
           A simple and efficient method for global sensitivity analysis 
           based on cumulative distribution functions. 
           Environmental Modelling & Software 67, 1–11. 
           https://doi.org/10.1016/j.envsoft.2015.01.004
    
    .. [3] Baroni, G., Francke, T., 2020. 
           An effective strategy for combining variance- and distribution-based 
           global sensitivity analysis. 
           Environmental Modelling & Software 134, 104851. 
           https://doi.org/10.1016/j.envsoft.2020.104851


    Examples
    --------
        >>> X = latin.sample(problem, 1000)
        >>> Y = Ishigami.evaluate(X)
        >>> Si = pawn.analyze(problem, X, Y)
    """
    if seed:
        np.random.seed(seed)

    groups = _check_groups(problem)
    if not groups:
        D = problem['num_vars']
    else:
        _, D = extract_group_names(groups)

    result = np.zeros((D, ))
    conf = result.copy()

    if isinstance(stat, str):
        try:
            stat_func = getattr(np, stat)
        except AttributeError:
            raise AttributeError(
                "Could not find specified statistic: '{}'".format(stat))
    elif callable(stat):
        stat_func = stat
    else:
        raise ValueError("Unknown statistic: '{}'".format(stat))

    step = (1/S)
    seq = np.arange(0.0, 1+step, step)
    for d_i in range(D):
        X_di = X[:, d_i]
        if X_di.ptp() == 0.0:
            # Input is constant
            result[d_i] = 0.0
            conf[d_i] = 0.0
            continue

        X_q = np.nanquantile(X_di, seq)

        result[d_i], Nc = _calc_ks(X_di, X_q, Y, S, stat_func)
        conf[d_i] = _bootstrap(X_di, seq, Y, Nc, S, num_resamples, conf_level, 
                               stat_func)

    Si = ResultDict([('PAWNi', result), ('PAWNi_conf', conf)])
    Si['names'] = problem['names']

    if print_to_console:
        print(Si.to_df())

    return Si


def _bootstrap(X_di, seq, Y, Nc, S, num_resamples, conf_level, stat_func):
    """Bootstrap without replacement.

    `Nc` does not need to be selected by the user, as it is simply the number
    of points in each interval (see [1] in `analyze`).

    Reason for preferring bootstrap without replacement is given in the
    supplementary materials of [1] (see reference list below).


    Parameters
    ----------
    Y : np.array
        Model outputs
    Nc : int
        Size of conditional sample
    num_resamples : int
        Number of bootstrap samples to take (< N, where N is number of available samples)
    conf_level : float
        The confidence interval level (default 0.95)


    References
    ----------
    .. [1] Khorashadi Zadeh, F., Nossent, J., Sarrazin, F., Pianosi, F.,
           van Griensven, A., Wagener, T., Bauwens, W., 2017.
           Comparison of variance-based and moment-independent global
           sensitivity analysis approaches by application to the SWAT
           model.
           Environmental Modelling & Software 91, 210–222.
           https://doi.org/10.1016/j.envsoft.2017.02.001

    """
    s = np.full(num_resamples, np.nan)
    Y_len = Y.shape[0]
    for i in range(num_resamples):
        # Random extraction from Y
        r = np.random.choice(Y_len, Nc, replace=False)
        Y_sel = Y[r]
        X_r = X_di[r]

        if (X_r.shape[0] == 0) or (X_r.ptp() == 0.0):
            # Input is constant
            s[i] = 0.0
            continue

        X_q = np.nanquantile(X_r, seq)
        s[i], _ = _calc_ks(X_r, X_q, Y_sel, S, stat_func)
    
    return norm.ppf(0.5 + conf_level / 2.0) * s.std(ddof=1)


def _calc_ks(X_r, X_q, Y_sel, S, stat_func):
    s = np.full(S, np.nan)
    for s_i in range(S):
        Ys = Y_sel[(X_r >= X_q[s_i]) & (X_r < X_q[s_i+1])]

        # kstest function returns a KS object which holds the KS statistic
        # and p-value
        # Note from documentation:
        # if the K-S statistic is small or the p-value is high, then 
        # we cannot reject the hypothesis that the distributions of 
        # the two samples are the same.
        try:
            s[s_i] = kstest(Ys, Y_sel).statistic
        except ValueError:
            # Ys is empty
            s[s_i] = 0.0

    return stat_func(s), Ys.shape[0]


def cli_parse(parser):
    parser.add_argument('-X', '--model-input-file',
                        type=str, required=True, help='Model input file')
    parser.add_argument('-S', '--slices',
                        type=int, required=False, default=10, help='Number of intervals to partition input-output space')
    parser.add_argument('-st', '--statistic',
                        type=str, required=False, default='median', help='Numpy compatible statistic to use (defaults to median)')
    parser.add_argument('-r', '--resamples', type=int, required=False,
                        default=100,
                        help='Number of bootstrap resamples to estimate'
                        'confidence intervals')
    return parser


def cli_action(args):
    problem = read_param_file(args.paramfile)
    X = np.loadtxt(args.model_input_file,
                   delimiter=args.delimiter)
    Y = np.loadtxt(args.model_output_file,
                   delimiter=args.delimiter,
                   usecols=(args.column,))
    analyze(problem, X, Y, S=args.slices, print_to_console=True, seed=args.seed)


if __name__ == "__main__":
    common_args.run_cli(cli_parse, cli_action)
