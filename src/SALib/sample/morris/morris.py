import numpy as np
from typing import Dict, Optional, Union

import warnings

from .local import LocalOptimisation
from .brute import BruteForce

from .strategy import SampleMorris

from SALib.sample import common_args
from SALib.util import (
    scale_samples,
    read_param_file,
    compute_groups_matrix,
    _define_problem_with_groups,
    _compute_delta,
    handle_seed,
)

__all__ = ["sample"]


def sample(
    problem: Dict,
    N: int,
    num_levels: int = 4,
    optimal_trajectories: Optional[Union[int, None]] = None,
    local_optimization: bool = True,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> np.ndarray:
    """Generate model inputs using the Method of Morris.

    Three variants of Morris' sampling for elementary effects is supported:

    - Vanilla Morris (see [1])
      when ``optimal_trajectories`` is ``None``/``False`` and
      ``local_optimization`` is ``False``
    - Optimised trajectories when ``optimal_trajectories=True`` using
        Campolongo's enhancements (see [2]) and optionally Ruano's enhancement
        (see [3]) when ``local_optimization=True``
    - Morris with groups when the problem definition specifies groups of
      parameters

    Results from these model inputs are intended to be used with
    :func:`SALib.analyze.morris.analyze`.

    Notes
    -----
    Campolongo et al., [2] introduces an optimal trajectories approach which
    attempts to maximize the parameter space scanned for a given number of
    trajectories (where `optimal_trajectories` :math:`\\in {2, ..., N}`).
    The approach accomplishes this aim by randomly generating a high number
    of possible trajectories (500 to 1000 in [2]) and selecting a subset of
    ``r`` trajectories which have the highest spread in parameter space.
    The ``r`` variable in [2] corresponds to the ``optimal_trajectories``
    parameter here.

    Calculating all possible combinations of trajectories can be
    computationally expensive. The number of factors makes little
    difference, but the ratio between number of optimal trajectories and the
    sample size results in an exponentially increasing number of scores that
    must be computed to find the optimal combination of trajectories. We
    suggest going no higher than 4 levels from a pool of 100 samples with this
    "brute force" approach.

    Ruano et al., [3] proposed an alternative approach with an iterative
    process that maximizes the distance between subgroups of generated
    trajectories, from which the final set of trajectories are selected, again
    maximizing the distance between each. The approach is not guaranteed to
    produce the most optimal spread of trajectories, but are at least locally
    maximized and significantly reduce the time taken to select trajectories.
    With ``local_optimization = True`` (which is default), it is possible to
    go higher than the previously suggested 4 levels from a pool of 100
    samples.

    Parameters
    ----------
    problem : dict
        The problem definition
    N : int
        The number of trajectories to generate
    num_levels : int, default=4
        The number of grid levels (should be even)
    optimal_trajectories : int
        The number of optimal trajectories to sample (between 2 and N)
    local_optimization : bool, default=True
        Flag whether to use local optimization according to Ruano et al. (2012)
        Speeds up the process tremendously for bigger N and num_levels.
        If set to ``False`` brute force method is used, unless ``gurobipy`` is
        available
    seed : {None, int, `numpy.random.Generator`}, optional
        If `seed` is None the `numpy.random.Generator` generator is used.
        If `seed` is an int, a new ``Generator`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` instance then that instance is
        used. Default is None.

    Returns
    -------
    sample_morris : np.ndarray
        Array containing the model inputs required for Method of Morris.
        The resulting matrix has :math:`(G/D+1)*N/T` rows and :math:`D`
        columns, where :math:`D` is the number of parameters, :math:`G`
        is the number of groups (if no groups are selected, the number
        of parameters). :math:`T` is the number of trajectories
        :math:`N`, or `optimal_trajectories` if selected.

    References
    ----------
    1. Morris, M.D., 1991.
       Factorial Sampling Plans for Preliminary Computational Experiments.
       Technometrics 33, 161-174.
       https://doi.org/10.1080/00401706.1991.10484804

    2. Campolongo, F., Cariboni, J., & Saltelli, A. 2007.
       An effective screening design for sensitivity analysis of large
       models.
       Environmental Modelling & Software, 22(10), 1509-1518.
       https://doi.org/10.1016/j.envsoft.2006.10.004

    3. Ruano, M.V., Ribes, J., Seco, A., Ferrer, J., 2012.
       An improved sampling strategy based on trajectory design for
       application of the Morris method to systems with many input
       factors.
       Environmental Modelling & Software 37, 103-109.
       https://doi.org/10.1016/j.envsoft.2012.03.008
    """
    rng = handle_seed(seed)

    _check_if_num_levels_is_even(num_levels)

    problem = _define_problem_with_groups(problem)

    sample_morris = _sample_morris(problem, N, rng, num_levels=num_levels)

    if optimal_trajectories:
        if local_optimization and (
            not isinstance(optimal_trajectories, int) or optimal_trajectories > N
        ):
            msg = "optimal_trajectories should be an " f"integer between 2 and {N}"
            raise ValueError(msg)

        sample_morris = _compute_optimised_trajectories(
            problem, sample_morris, N, optimal_trajectories, local_optimization
        )

    sample_morris = scale_samples(sample_morris, problem)

    return sample_morris


def _sample_morris(
    problem: Dict,
    number_trajectories: int,
    rng: np.random.Generator,
    num_levels: int = 4,
) -> np.ndarray:
    """Generate trajectories for groups

    Returns an :math:`N(g+1)`-by-:math:`k` array of `N` trajectories,
    where :math:`g` is the number of groups and :math:`k` is the number
    of factors

    Parameters
    ---------
    problem : dict
        The problem definition
    number_trajectories : int
        The number of trajectories to generate
    rng : np.random.Generator
    num_levels : int, default=4
        The number of grid levels

    Returns
    -------
    numpy.ndarray
    """
    group_membership, _ = compute_groups_matrix(problem["groups"])
    _check_group_membership(group_membership)

    num_params = group_membership.shape[0]
    num_groups = group_membership.shape[1]

    sample_morris = [
        _generate_trajectory(group_membership, rng, num_levels=num_levels)
        for _ in range(number_trajectories)
    ]
    sample_morris = np.array(sample_morris)

    return sample_morris.reshape((number_trajectories * (num_groups + 1), num_params))


def _generate_trajectory(
    group_membership: np.ndarray, rng: np.random.Generator, num_levels: int = 4
) -> np.ndarray:
    """Return a single trajectory

    Return a single trajectory of size :math:`(g+1)`-by-:math:`k`
    where :math:`g` is the number of groups,
    and :math:`k` is the number of factors,
    both implied by the dimensions of `group_membership`

    Parameters
    ---------
    group_membership : np.ndarray
        a k-by-g matrix which notes factor membership of groups
    rng : numpy.random.Generator
    num_levels : int, default=4
        The number of levels in the grid

    Returns
    -------
    np.ndarray
    """
    delta = _compute_delta(num_levels)

    # Infer number of groups `g` and number of params `k` from
    # `group_membership` matrix
    num_params = group_membership.shape[0]
    num_groups = group_membership.shape[1]

    # Matrix B - size (g + 1) * g -  lower triangular matrix
    B = np.tril(np.ones([num_groups + 1, num_groups], dtype=int), -1)

    P_star = _generate_p_star(num_groups, rng)

    # Matrix J - a (g+1)-by-num_params matrix of ones
    J = np.ones((num_groups + 1, num_params))

    # Matrix D* - num_params-by-num_params matrix which describes whether
    # factors move up or down
    D_star = np.diag(rng.choice([-1, 1], num_params))

    x_star = _generate_x_star(num_params, num_levels, rng)

    # Matrix B* - size (num_groups + 1) * num_params
    B_star = _compute_b_star(J, x_star, delta, B, group_membership, P_star, D_star)

    return B_star


def _compute_b_star(
    J: np.ndarray,
    x_star: np.ndarray,
    delta: float,
    B: np.ndarray,
    G: np.ndarray,
    P_star: np.ndarray,
    D_star: np.ndarray,
) -> np.ndarray:
    """
    Compute the random sampling matrix B*.

    Parameters
    ----------
    J: matrix of 1's
    x_star: randomly chosen "base value" of x
    delta: parameters variation
    B: sampling matrix (not random)
    G: groups matrix
    P_star: random permutation matrix
    D_star: diagonal matrix with each element being +1 or -1, with equal
            probability

    Returns
    -------
    Random sampling matrix B*
    """
    element_a = J[0, :] * x_star
    element_b = np.matmul(G, P_star).T
    element_c = np.matmul(2.0 * B, element_b)
    element_d = np.matmul((element_c - J), D_star)

    b_star = element_a + (delta / 2.0) * (element_d + J)
    return b_star


def _generate_p_star(num_groups: int, rng: np.random.Generator) -> np.ndarray:
    """Describe the order in which groups move

    Parameters
    ---------
    num_groups : int
    rng : numpy.random.Generator

    Returns
    -------
    np.ndarray
        Matrix P* - size (g-by-g)
    """
    p_star = np.eye(num_groups, num_groups)
    rng.shuffle(p_star)
    return p_star


def _generate_x_star(
    num_params: int, num_levels: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate an 1-by-num_params array to represent initial position for EE

    This should be a randomly generated array in the p level grid
    :math:`\\omega`

    Parameters
    ---------
    num_params : int
        The number of parameters (factors)
    num_levels : int
        The number of levels
    rng : numpy.random.Generator

    Returns
    -------
    numpy.ndarray
        The initial starting positions of the trajectory
    """
    x_star = np.zeros((1, num_params))
    delta = _compute_delta(num_levels)
    bound = 1 - delta
    grid = np.linspace(0, bound, int(num_levels / 2))

    x_star[0, :] = rng.choice(grid, num_params)

    return x_star


def _compute_optimised_trajectories(
    problem: Dict,
    input_sample: int,
    N: int,
    k_choices: int,
    local_optimization: bool = False,
) -> np.ndarray:
    """
    Calls the procedure to compute the optimum k_choices of trajectories
    from the input_sample.
    If there are groups, then this procedure allocates the groups to the
    correct call here.

    Parameters
    ---------
    problem : dict
        The problem definition
    input_sample :
    N : int
        The number of samples to generate
    k_choices : int
        The number of optimal trajectories
    local_optimization : bool, default=False
        If true, uses local optimisation heuristic
    """
    if local_optimization is False and k_choices > 10:
        msg = "Running optimal trajectories greater than values of 10 \
                will take a long time."
        raise ValueError(msg)

    if np.any((input_sample < 0) | (input_sample > 1)):
        raise ValueError("Input sample must be scaled between 0 and 1")

    num_groups = len(set(problem["groups"]))
    num_params = problem["num_vars"]

    strategy = _choose_optimization_strategy(local_optimization)
    context = SampleMorris(strategy)

    output = context.sample(input_sample, N, num_params, k_choices, num_groups)

    return output


def _check_if_num_levels_is_even(num_levels: int):
    """
    Checks if the number of levels is even. If not, raises a warn.

    Parameters
    ----------
    num_levels: int
        Number of levels
    """
    if not num_levels % 2 == 0:
        warnings.warn("num_levels should be an even number, " "sample may be biased")


def _check_group_membership(group_membership: np.ndarray):
    """
    Checks if the group_membership matrix was correctly defined

    Parameters
    ----------
    group_membership: np.array
        Group membership matrix
    """
    if group_membership is None:
        raise ValueError("Please define the 'group_membership' matrix")
    if not isinstance(group_membership, np.ndarray):
        raise TypeError(
            "Argument 'group_membership' should be formatted \
                         as a numpy np.ndarray"
        )


def _choose_optimization_strategy(local_optimization: bool):
    """
    Choose the strategy to optimize the trajectories.

    Parameters
    ----------
    local_optimization: boolean to indicate if a local optimization should be
                        used.

    Returns
    -------

    """
    if local_optimization:
        # Use local method
        strategy = LocalOptimisation()
    else:
        # Use brute force approach
        strategy = BruteForce()

    return strategy


def cli_parse(parser):
    parser.add_argument(
        "-l",
        "--levels",
        type=int,
        required=False,
        default=4,
        help="Number of grid levels \
                        (Morris only)",
    )
    parser.add_argument(
        "-k",
        "--k-optimal",
        type=int,
        required=False,
        default=None,
        help="Number of optimal trajectories \
                        (Morris only)",
    )
    parser.add_argument(
        "-lo",
        "--local",
        type=bool,
        required=True,
        default=False,
        help="Use the local optimisation method \
                        (Morris with optimization only)",
    )
    return parser


def cli_action(args):
    problem = read_param_file(args.paramfile)
    param_values = sample(
        problem,
        args.samples,
        num_levels=args.levels,
        optimal_trajectories=args.k_optimal,
        local_optimization=args.local,
        seed=args.seed,
    )

    np.savetxt(
        args.output,
        param_values,
        delimiter=args.delimiter,
        fmt="%." + str(args.precision) + "e",
    )


if __name__ == "__main__":
    common_args.run_cli(cli_parse, cli_action)
