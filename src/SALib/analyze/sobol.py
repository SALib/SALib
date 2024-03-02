from types import MethodType
from warnings import warn

from scipy.stats import norm

import numpy as np
import pandas as pd

from . import common_args
from ..util import read_param_file, ResultDict, extract_group_names, _check_groups

from multiprocessing import Pool, cpu_count
from functools import partial
from itertools import combinations, zip_longest


CONST_RESULT_MSG = (
    "Constant values encountered, indicating model evaluations "
    "(or subset of evaluations) produced identical values."
)


def analyze(
    problem,
    Y,
    calc_second_order=True,
    num_resamples=100,
    conf_level=0.95,
    print_to_console=False,
    parallel=False,
    n_processors=None,
    keep_resamples=False,
    seed=None,
):
    """Perform Sobol Analysis on model outputs.

    Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf', where
    each entry is a list of size D (the number of parameters) containing the
    indices in the same order as the parameter file.  If calc_second_order is
    True, the dictionary also contains keys 'S2' and 'S2_conf'.

    There are several approaches to estimating sensitivity indices.
    The general approach is described in [1]. The implementation offered here
    follows [2] for first and total order indices, whereas estimation of
    second order sensitivities follows [3]. A noteworthy point is the
    improvement to reduce error rates in sensitivity estimation is introduced
    in [4].

    Notes
    -----
    Compatible with:
        `saltelli` : :func:`SALib.sample.saltelli.sample`
        `sobol` : :func:`SALib.sample.sobol.sample`


    Examples
    --------
    >>> X = saltelli.sample(problem, 512)
    >>> Y = Ishigami.evaluate(X)
    >>> Si = sobol.analyze(problem, Y, print_to_console=True)


    Parameters
    ----------
    problem : dict
        The problem definition
    Y : numpy.array
        A NumPy array containing the model outputs
    calc_second_order : bool
        Calculate second-order sensitivities (default True)
    num_resamples : int
        The number of resamples (default 100)
    conf_level : float
        The confidence interval level (default 0.95)
    print_to_console : bool
        Print results directly to console (default False)
    parallel : bool
        Perform analysis in parallel if True
    n_processors : int
        Number of parallel processes (only used if parallel is True)
    keep_resamples : bool
        Whether or not to store intermediate resampling results (default False)
    seed : int
        Seed to generate a random number

    References
    ----------
    1. Sobol, I. M. (2001).
       Global sensitivity indices for nonlinear
       mathematical models and their Monte Carlo estimates.
       Mathematics and Computers in Simulation, 55(1-3):271-280,
       doi:10.1016/S0378-4754(00)00270-6.

    2. Saltelli, A., P. Annoni, I. Azzini, F. Campolongo, M. Ratto, and
       S. Tarantola (2010).
       Variance based sensitivity analysis of model output. Design and
       estimator for the total sensitivity index.
       Computer Physics Communications, 181(2):259-270,
       doi:10.1016/j.cpc.2009.09.018.

    3. Saltelli, A. (2002).
       Making best use of model evaluations to compute sensitivity indices.
       Computer Physics Communications, 145(2):280-297
       doi:10.1016/S0010-4655(02)00280-1.

    4. Sobol', I. M., Tarantola, S., Gatelli, D., Kucherenko, S. S., &
       Mauntz, W. (2007).
       Estimating the approximation error when fixing unessential factors in global
       sensitivity analysis.
       Reliability Engineering & System Safety, 92(7), 957-960.
       https://doi.org/10.1016/j.ress.2006.07.001
    """
    if seed:
        # Set seed to ensure CIs are the same
        rng = np.random.default_rng(seed).integers
    else:
        rng = np.random.randint

    # determining if groups are defined and adjusting the number
    # of rows in the cross-sampled matrix accordingly
    groups = _check_groups(problem)
    if not groups:
        D = problem["num_vars"]
    else:
        _, D = extract_group_names(groups)

    if calc_second_order and Y.size % (2 * D + 2) == 0:
        N = int(Y.size / (2 * D + 2))
    elif not calc_second_order and Y.size % (D + 2) == 0:
        N = int(Y.size / (D + 2))
    else:
        raise RuntimeError(
            """
        Incorrect number of samples in model output file.
        Confirm that calc_second_order matches option used during sampling."""
        )

    if not 0 < conf_level < 1:
        raise RuntimeError("Confidence level must be between 0-1.")

    # Normalize the model output.
    # Estimates of the Sobol' indices can be biased for non-centered outputs
    # so we center here by normalizing with the standard deviation.
    # Other approaches opt to subtract the mean.
    Y = (Y - Y.mean()) / Y.std()

    A, B, AB, BA = separate_output_values(Y, D, N, calc_second_order)
    r = rng(N, size=(N, num_resamples))
    Z = norm.ppf(0.5 + conf_level / 2)

    if not parallel:
        S = create_Si_dict(D, num_resamples, keep_resamples, calc_second_order)

        for j in range(D):
            S["S1"][j] = first_order(A, AB[:, j], B)
            S1_conf_j = first_order(A[r], AB[r, j], B[r])

            if keep_resamples:
                S["S1_conf_all"][:, j] = S1_conf_j

            var_diff = np.r_[A[r], B[r]].ptp()
            if var_diff != 0.0:
                S["S1_conf"][j] = Z * S1_conf_j.std(ddof=1)
            else:
                S["S1_conf"][j] = 0.0

            S["ST"][j] = total_order(A, AB[:, j], B)

            ST_conf_j = total_order(A[r], AB[r, j], B[r])

            if keep_resamples:
                S["ST_conf_all"][:, j] = ST_conf_j

            if var_diff != 0.0:
                S["ST_conf"][j] = Z * ST_conf_j.std(ddof=1)
            else:
                S["ST_conf"][j] = 0.0

        # Second order (+conf.)
        if calc_second_order:
            for j in range(D):
                for k in range(j + 1, D):
                    S["S2"][j, k] = second_order(A, AB[:, j], AB[:, k], BA[:, j], B)
                    S["S2_conf"][j, k] = Z * second_order(
                        A[r], AB[r, j], AB[r, k], BA[r, j], B[r]
                    ).std(ddof=1)
    else:
        tasks, n_processors = create_task_list(D, calc_second_order, n_processors)

        func = partial(sobol_parallel, Z, A, AB, BA, B, r)
        pool = Pool(n_processors)
        S_list = pool.map_async(func, tasks)
        pool.close()
        pool.join()

        S = Si_list_to_dict(
            S_list.get(), D, num_resamples, keep_resamples, calc_second_order
        )

    # Add problem context and override conversion method for special case
    S.problem = problem
    S.to_df = MethodType(to_df, S)

    # Print results to console
    if print_to_console:
        res = S.to_df()
        for df in res:
            print(df)

    return S


def first_order(A, AB, B):
    """
    First order estimator following Saltelli et al. 2010 CPC, normalized by
    sample variance
    """
    y = np.r_[A, B]
    if y.ptp() == 0:
        warn(CONST_RESULT_MSG)
        return np.array([0.0])

    return np.mean(B * (AB - A), axis=0) / np.var(y, axis=0)


def total_order(A, AB, B):
    """
    Total order estimator following Saltelli et al. 2010 CPC, normalized by
    sample variance
    """
    y = np.r_[A, B]
    if y.ptp() == 0:
        warn(CONST_RESULT_MSG)
        return np.array([0.0])

    return 0.5 * np.mean((A - AB) ** 2, axis=0) / np.var(y, axis=0)


def second_order(A, ABj, ABk, BAj, B):
    """Second order estimator following Saltelli 2002"""
    y = np.r_[A, B]
    if y.ptp() == 0:
        warn(CONST_RESULT_MSG)
        return np.array([0.0])

    Vjk = np.mean(BAj * ABk - A * B, axis=0) / np.var(y, axis=0)
    Sj = first_order(A, ABj, B)
    Sk = first_order(A, ABk, B)

    return Vjk - Sj - Sk


def create_Si_dict(
    D: int, num_resamples: int, keep_resamples: bool, calc_second_order: bool
):
    """initialize empty dict to store sensitivity indices"""
    S = ResultDict((k, np.zeros(D)) for k in ("S1", "S1_conf", "ST", "ST_conf"))

    if keep_resamples:
        # Create entries to store intermediate resampling results
        S["S1_conf_all"] = np.zeros((num_resamples, D))
        S["ST_conf_all"] = np.zeros((num_resamples, D))

    if calc_second_order:
        S["S2"] = np.full((D, D), np.nan)
        S["S2_conf"] = np.full((D, D), np.nan)

    return S


def separate_output_values(Y, D, N, calc_second_order):
    AB = np.zeros((N, D))
    BA = np.zeros((N, D)) if calc_second_order else None
    step = 2 * D + 2 if calc_second_order else D + 2

    A = Y[0 : Y.size : step]
    B = Y[(step - 1) : Y.size : step]
    for j in range(D):
        AB[:, j] = Y[(j + 1) : Y.size : step]
        if calc_second_order:
            BA[:, j] = Y[(j + 1 + D) : Y.size : step]

    return A, B, AB, BA


def sobol_parallel(Z, A, AB, BA, B, r, tasks):
    sobol_indices = []
    for d, j, k in tasks:
        if d == "S1":
            s = first_order(A, AB[:, j], B)
        elif d == "S1_conf":
            s = Z * first_order(A[r], AB[r, j], B[r]).std(ddof=1)
        elif d == "ST":
            s = total_order(A, AB[:, j], B)
        elif d == "ST_conf":
            s = Z * total_order(A[r], AB[r, j], B[r]).std(ddof=1)
        elif d == "S2":
            s = second_order(A, AB[:, j], AB[:, k], BA[:, j], B)
        elif d == "S2_conf":
            s = Z * second_order(A[r], AB[r, j], AB[r, k], BA[r, j], B[r]).std(ddof=1)
        sobol_indices.append([d, j, k, s])

    return sobol_indices


def create_task_list(D, calc_second_order, n_processors):
    """
    Create list with one entry (key, parameter 1, parameter 2) per sobol
    index (+conf.). This is used to supply parallel tasks to
    multiprocessing.Pool
    """
    tasks_first_order = [
        [d, j, None] for j in range(D) for d in ("S1", "S1_conf", "ST", "ST_conf")
    ]

    # Add second order (+conf.) to tasks
    tasks_second_order = []
    if calc_second_order:
        tasks_second_order = [
            [d, j, k]
            for j in range(D)
            for k in range(j + 1, D)
            for d in ("S2", "S2_conf")
        ]

    if n_processors is None:
        n_processors = min(
            cpu_count(), len(tasks_first_order) + len(tasks_second_order)
        )

    if not calc_second_order:
        tasks = np.array_split(tasks_first_order, n_processors)
    else:
        # merges both lists alternating its elements and splits the
        # resulting lists into n_processors sublists
        tasks = np.array_split(
            [
                v
                for v in sum(
                    zip_longest(tasks_first_order[::-1], tasks_second_order), ()
                )
                if v is not None
            ],
            n_processors,
        )

    return tasks, n_processors


def Si_list_to_dict(
    S_list, D: int, num_resamples: int, keep_resamples: bool, calc_second_order: bool
):
    """Convert the parallel output into the regular dict format for
    printing/returning"""
    S = create_Si_dict(D, num_resamples, keep_resamples, calc_second_order)
    L = []
    for list in S_list:  # first reformat to flatten
        L += list

    for s in L:  # First order (+conf.)
        if s[2] is None:
            S[s[0]][s[1]] = s[3]
        else:
            S[s[0]][s[1], s[2]] = s[3]

    return S


def Si_to_pandas_dict(S_dict):
    """Convert Si information into Pandas DataFrame compatible dict.


    Examples
    --------
    >>> X = saltelli.sample(problem, 512)
    >>> Y = Ishigami.evaluate(X)
    >>> Si = sobol.analyze(problem, Y, print_to_console=True)
    >>> T_Si, first_Si, (idx, second_Si) = sobol.Si_to_pandas_dict(Si, problem)


    Parameters
    ----------
    S_dict : ResultDict
        Sobol sensitivity indices

    See Also
    --------
    Si_list_to_dict

    Returns
    -------
    tuple : of total, first, and second order sensitivities.
            Total and first order are dicts.
            Second order sensitivities contain a tuple of parameter name
            combinations for use as the DataFrame index and second order
            sensitivities.
            If no second order indices found, then returns tuple of
            (None, None)
    """
    problem = S_dict.problem
    total_order = {"ST": S_dict["ST"], "ST_conf": S_dict["ST_conf"]}
    first_order = {"S1": S_dict["S1"], "S1_conf": S_dict["S1_conf"]}

    idx = None
    second_order = None
    if "S2" in S_dict:
        groups = _check_groups(problem)
        if groups:
            names, _ = extract_group_names(groups)
        else:
            names = problem.get("names")

        if len(names) > 2:
            idx = list(combinations(names, 2))
        else:
            idx = (names,)

        second_order = {
            "S2": [S_dict["S2"][names.index(i[0]), names.index(i[1])] for i in idx],
            "S2_conf": [
                S_dict["S2_conf"][names.index(i[0]), names.index(i[1])] for i in idx
            ],
        }
    return total_order, first_order, (idx, second_order)


def to_df(self):
    """Conversion method to Pandas DataFrame. To be attached to ResultDict.

    Returns
    -------
    List : of Pandas DataFrames in order of Total, First, Second

    Examples
    --------
    >>> Si = sobol.analyze(problem, Y, print_to_console=True)
    >>> total_Si, first_Si, second_Si = Si.to_df()
    """
    total, first, (idx, second) = Si_to_pandas_dict(self)

    problem = self.problem
    groups = _check_groups(problem)
    if not groups:
        names = problem.get("names")
    else:
        names, _ = extract_group_names(groups)

    ret = [pd.DataFrame(total, index=names), pd.DataFrame(first, index=names)]

    if second:
        ret += [pd.DataFrame(second, index=idx)]

    return ret


def cli_parse(parser):
    parser.add_argument(
        "--max-order",
        type=int,
        required=False,
        default=2,
        choices=[1, 2],
        help="Maximum order of sensitivity indices to " "calculate",
    )
    parser.add_argument(
        "-r",
        "--resamples",
        type=int,
        required=False,
        default=100,
        help="Number of bootstrap resamples for Sobol " "confidence intervals",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Makes " "use of parallelization.",
        dest="parallel",
    )
    parser.add_argument(
        "--processors",
        type=int,
        required=False,
        default=None,
        help="Number of processors to be used with the " + "parallel option.",
        dest="n_processors",
    )
    return parser


def cli_action(args):
    problem = read_param_file(args.paramfile)
    Y = np.loadtxt(
        args.model_output_file, delimiter=args.delimiter, usecols=(args.column,)
    )

    analyze(
        problem,
        Y,
        (args.max_order == 2),
        num_resamples=args.resamples,
        print_to_console=True,
        parallel=args.parallel,
        n_processors=args.n_processors,
        seed=args.seed,
    )


if __name__ == "__main__":
    common_args.run_cli(cli_parse, cli_action)
