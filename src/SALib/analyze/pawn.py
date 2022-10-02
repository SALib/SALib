from typing import Dict

import numpy as np
from scipy.stats import ks_2samp

from . import common_args
from ..util import read_param_file, ResultDict, extract_group_names, _check_groups


def analyze(
    problem: Dict,
    X: np.ndarray,
    Y: np.ndarray,
    S: int = 10,
    print_to_console: bool = False,
    seed: int = None,
):
    """Performs PAWN sensitivity analysis.

    The PAWN method [1] is a moment-independent approach to Global Sensitivity
    Analysis (GSA). It is described as producing robust results at relatively
    low sample sizes (see [2]) for the purpose of factor ranking and screening.

    The distribution of model outputs is examined rather than
    their variation as is typical in other common GSA approaches. The PAWN
    method further distinguishes itself from other moment-independent
    approaches by characterizing outputs by their cumulative distribution
    function (CDF) as opposed to their probability distribution function.
    As the CDF for a given random variable is typically normally distributed,
    PAWN can be more appropriately applied when outputs are highly-skewed or
    multi-modal, for which variance-based methods may produce unreliable
    results.

    PAWN characterizes the relationship between inputs and outputs by
    quantifying the variation in the output distributions after conditioning
    an input. A factor is deemed non-influential if distributions coincide at
    all ``S`` conditioning intervals. The Kolmogorov-Smirnov statistic is used
    as a measure of distance between the distributions.

    This implementation reports the PAWN index at the min, mean, median, and
    max across the slides/conditioning intervals as well as the coefficient of
    variation (``CV``). The median value is the typically reported value. As
    the ``CV`` is (standard deviation / mean), it indicates the level of
    variability across the slides, with values closer to zero indicating lower
    variation.


    Notes
    -----
    Compatible with:
        all samplers

    This implementation ignores all NaNs.

    When applied to grouped factors, the analysis is conducted on each factor
    individually, and the mean of their results are reported.


    Examples
    --------
        >>> X = latin.sample(problem, 1000)
        >>> Y = Ishigami.evaluate(X)
        >>> Si = pawn.analyze(problem, X, Y, S=10, print_to_console=False)


    Parameters
    ----------
    problem : dict
        The problem definition
    X : numpy.array
        A NumPy array containing the model inputs
    Y : numpy.array
        A NumPy array containing the model outputs
    S : int
        Number of slides; the conditioning intervals (default 10)
    print_to_console : bool
        Print results directly to console (default False)
    seed : int
        Seed value to ensure deterministic results


    References
    ----------
    .. [1] Pianosi, F., Wagener, T., 2015.
           A simple and efficient method for global sensitivity analysis
           based on cumulative distribution functions.
           Environmental Modelling & Software 67, 1-11.
           https://doi.org/10.1016/j.envsoft.2015.01.004

    .. [2] Pianosi, F., Wagener, T., 2018.
           Distribution-based sensitivity analysis from a generic input-output sample.
           Environmental Modelling & Software 108, 197-207.
           https://doi.org/10.1016/j.envsoft.2018.07.019

    .. [3] Baroni, G., Francke, T., 2020.
           An effective strategy for combining variance- and
           distribution-based global sensitivity analysis.
           Environmental Modelling & Software, 134, 104851.
           https://doi.org/10.1016/j.envsoft.2020.104851

    .. [4] Baroni, G., Francke, T., 2020.
           GSA-cvd
           Combining variance- and distribution-based global sensitivity analysis
           https://github.com/baronig/GSA-cvd
    """
    if seed:
        np.random.seed(seed)

    D = problem["num_vars"]
    groups = _check_groups(problem)
    if not groups:
        var_names = problem["names"]
    else:
        var_names, _ = extract_group_names(problem.get("groups", []))

    results = np.full((D, 5), np.nan)
    temp_pawn = np.full((S, D), np.nan)

    step = 1 / S
    for d_i in range(D):
        seq = np.arange(0, 1 + step, step)
        X_di = X[:, d_i]
        X_q = np.nanquantile(X_di, seq)

        for s in range(S):
            Y_sel = Y[(X_di >= X_q[s]) & (X_di < X_q[s + 1])]
            if len(Y_sel) == 0:
                # no available samples
                continue

            # KD value
            # Function returns a KS object which holds the KS statistic
            # and p-value
            # Note from scipy documentation:
            # if the K-S statistic is small or the p-value is high, then
            # we cannot reject the hypothesis that the distributions of
            # the two samples are the same.
            ks = ks_2samp(Y_sel, Y)
            temp_pawn[s, d_i] = ks.statistic

        p_ind = temp_pawn[:, d_i]
        mins = np.nanmin(p_ind)
        mean = np.nanmean(p_ind)
        med = np.nanmedian(p_ind)
        maxs = np.nanmax(p_ind)
        cv = np.nanstd(p_ind) / mean
        results[d_i, :] = [mins, mean, med, maxs, cv]

    if groups:
        groups = np.array(groups)
        unique_grps = [*dict.fromkeys(groups)]
        tmp = np.full((len(unique_grps), 5), np.nan)

        # Take the mean of effects from parameters that are grouped together
        for grp_id, grp in enumerate(unique_grps):
            tmp[grp_id, :] = np.mean(results[groups == grp, :], axis=0)

        results = tmp
        tmp = None

    Si = ResultDict(
        [
            ("minimum", results[:, 0]),
            ("mean", results[:, 1]),
            ("median", results[:, 2]),
            ("maximum", results[:, 3]),
            ("CV", results[:, 4]),
        ]
    )
    Si["names"] = var_names

    if print_to_console:
        print(Si.to_df())

    return Si


def cli_parse(parser):
    parser.add_argument(
        "-X", "--model-input-file", type=str, required=True, help="Model input file"
    )

    parser.add_argument(
        "-S", "--slices", type=int, required=False, help="Number of slices to take"
    )
    return parser


def cli_action(args):
    problem = read_param_file(args.paramfile)
    X = np.loadtxt(args.model_input_file, delimiter=args.delimiter)
    Y = np.loadtxt(
        args.model_output_file, delimiter=args.delimiter, usecols=(args.column,)
    )
    analyze(problem, X, Y, S=args.slices, print_to_console=True, seed=args.seed)


if __name__ == "__main__":
    common_args.run_cli(cli_parse, cli_action)
