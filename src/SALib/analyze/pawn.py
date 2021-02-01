import numpy as np
from scipy.stats import ks_2samp

from . import common_args

from ..util import (read_param_file, ResultDict, 
                    extract_group_names, _check_groups)


def analyze(problem, X, Y, S=10, print_to_console=False, seed=None):
    """Performs PAWN sensitivity analysis.

    Ported from an implementation in `R` by Baroni & Francke (2020)
    https://github.com/baronig/GSA-cvd

    Calculates the min, mean, median, max, and coefficient of variation (CV).

    CV is (standard deviation / mean), and so lower values indicate little change
    over the slides, and larger values indicate large variations across the slides.

    Parameters
    ----------
    problem : dict
        The problem definition
    X : numpy.array
        A NumPy array containing the model inputs
    Y : numpy.array
        A NumPy array containing the model outputs
    S : int
        Number of slides (default 10)
    print_to_console : bool
        Print results directly to console (default False)
    seed : int
        Seed value to ensure deterministic results

    References
    ----------
    .. [1] Pianosi, F., Wagener, T., 2015. 
           A simple and efficient method for global sensitivity analysis 
           based on cumulative distribution functions. 
           Environmental Modelling & Software 67, 1–11. 
           https://doi.org/10.1016/j.envsoft.2015.01.004


    Examples
    --------
    >>> X = latin.sample(problem, 1000)
    >>> Y = Ishigami.evaluate(X)
    >>> Si = pawn.analyze(problem, X, Y, S=10, print_to_console=False)
    """
    if seed:
        np.random.seed(seed)

    groups = _check_groups(problem)
    if not groups:
        D = problem['num_vars']
        var_names = problem['names']
    else:
        var_names, D = extract_group_names(problem.get('groups'))

    results = np.full((D, 5), np.nan)
    temp_pawn = np.full((S, D), np.nan)

    step = (1/S)
    for d_i in range(D):
        seq = np.arange(0, 1+step, step)
        X_q = np.nanquantile(X[:, d_i], seq)

        for s in range(S):
            Y_sel = Y[(X[:, d_i] >= X_q[s]) & (X[:, d_i] < X_q[s+1])]

            # KD value
            # Function returns a KS object which holds the KS statistic
            # and p-value
            # Note from documentation:
            # if the K-S statistic is small or the p-value is high, then 
            # we cannot reject the hypothesis that the distributions of 
            # the two samples are the same.
            ks = ks_2samp(Y_sel, Y)
            temp_pawn[s, d_i] = ks.statistic 

        p_ind = temp_pawn[:, d_i]
        mins = np.min(p_ind)
        mean = np.mean(p_ind)
        med = np.median(p_ind)
        maxs = np.max(p_ind)
        cv = np.std(p_ind) / mean
        results[d_i] = [mins, mean, med, maxs, cv]

    Si = ResultDict([
        ('minimum', results[:, 0]),
        ('mean', results[:, 1]),
        ('median', results[:, 2]),
        ('maximum', results[:, 3]),
        ('CV', results[:, 4]),
    ])
    Si['names'] = var_names

    if print_to_console:
        print(Si.to_df())

    return Si


def cli_parse(parser):
    parser.add_argument('-X', '--model-input-file',
                        type=str, required=True, help='Model input file')

    parser.add_argument('-S', '--slices',
                        type=int, required=False, help='Number of slices to take')
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
