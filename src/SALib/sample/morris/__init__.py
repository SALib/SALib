"""
Generate a sample using the Method of Morris

Three variants of Morris' sampling for elementary effects is supported:

- Vanilla Morris
- Optimised trajectories when ``optimal_trajectories=True`` (using
    Campolongo's enhancements from 2007 and optionally Ruano's enhancement
    from 2012; ``local_optimization=True``)
- Groups with optimised trajectories when ``optimal_trajectories=True`` and
    the problem definition specifies groups (note that ``local_optimization``
    must be ``False``)

At present, optimised trajectories is implemented using either a brute-force
approach, which can be very slow, especially if you require more than four
trajectories, or a local method based which is much faster. Both methods now
implement working with groups of factors.

Note that the number of factors makes little difference,
but the ratio between number of optimal trajectories and the sample size
results in an exponentially increasing number of scores that must be
computed to find the optimal combination of trajectories.  We suggest going
no higher than 4 from a pool of 100 samples with the brute force approach.
With local_optimization = True (which is default),
it is possible to go higher than the previously suggested 4 from 100.

"""
from __future__ import division

import numpy as np  # type: ignore
from typing import Dict

import numpy.random as rd  # type: ignore
import warnings

from . gurobi import GlobalOptimisation
from . local import LocalOptimisation
from . brute import BruteForce

from . strategy import SampleMorris

from SALib.sample import common_args
from SALib.util import scale_samples,nonuniform_scale_samples, read_param_file, compute_groups_matrix

try:
    import gurobipy  # type: ignore
except ImportError:
    _has_gurobi = False
else:
    _has_gurobi = True

__all__ = ['sample']


def sample(problem: Dict, N: int, num_levels: int=4, optimal_trajectories: int=None,
           local_optimization: bool=True, seed=None) -> np.array:
    """Generate model inputs using the Method of Morris

    Returns a NumPy matrix containing the model inputs required for Method of
    Morris.  The resulting matrix has :math:`(G+1)*T` rows and :math:`D`
    columns, where :math:`D` is the number of parameters, :math:`G` is the
    number of groups (if no groups are selected, the number of parameters).
    :math:`T` is the number of trajectories :math:`N`,
    or `optimal_trajectories` if selected.
    These model inputs  are intended to be used with
    :func:`SALib.analyze.morris.analyze`.

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
    seed : int, default=None

    Returns
    -------
    sample : numpy.ndarray
        Returns a numpy.ndarray containing the model inputs required for Method
        of Morris. The resulting matrix has :math:`(G/D+1)*N/T` rows and
        :math:`D` columns, where :math:`D` is the number of parameters.
    """
    if seed:
        np.random.seed(seed)

    if not num_levels % 2 == 0:
        warnings.warn("num_levels should be an even number, sample may be biased")
    if problem.get('groups'):
        sample = _sample_groups(problem, N, num_levels)
    else:
        sample = _sample_oat(problem, N, num_levels)

    if optimal_trajectories:

        sample = _compute_optimised_trajectories(problem,
                                                 sample,
                                                 N,
                                                 optimal_trajectories,
                                                 local_optimization)

    if not problem.get('dists'):
        # scaling values out of 0-1 range with uniform distributions
        scale_samples(sample, problem['bounds'])
        return sample
    else:
        # scaling values to other distributions based on inverse CDFs
        scaled_samples = nonuniform_scale_samples(sample, problem['bounds'], problem['dists'])
        return scaled_samples

def _sample_oat(problem: Dict, N: int, num_levels: int=4) -> np.ndarray:
    """Generate trajectories without groups

    Arguments
    ---------
    problem : dict
        The problem definition
    N : int
        The number of samples to generate
    num_levels : int, default=4
        The number of grid levels
    """
    group_membership = np.identity(problem['num_vars'], dtype=int)

    num_params = group_membership.shape[0]
    sample = np.array([generate_trajectory(group_membership,
                                           num_levels)
                       for n in range(N)])
    return sample.reshape((N * (num_params + 1), num_params))


def _sample_groups(problem, N, num_levels=4):
    """Generate trajectories for groups

    Returns an :math:`N(g+1)`-by-:math:`k` array of `N` trajectories,
    where :math:`g` is the number of groups and :math:`k` is the number
    of factors

    Arguments
    ---------
    problem : dict
        The problem definition
    N : int
        The number of trajectories to generate
    num_levels : int, default=4
        The number of grid levels

    Returns
    -------
    numpy.ndarray
    """
    if len(problem['groups']) != problem['num_vars']:
        raise ValueError("Groups do not match to number of variables")

    group_membership, _ = compute_groups_matrix(problem['groups'])

    if group_membership is None:
        raise ValueError("Please define the 'group_membership' matrix")
    if not isinstance(group_membership, np.ndarray):
        raise TypeError("Argument 'group_membership' should be formatted \
                         as a numpy ndarray")

    num_params = group_membership.shape[0]
    num_groups = group_membership.shape[1]
    sample = np.zeros((N * (num_groups + 1), num_params))
    sample = np.array([generate_trajectory(group_membership,
                                           num_levels)
                       for n in range(N)])
    return sample.reshape((N * (num_groups + 1), num_params))


def generate_trajectory(group_membership, num_levels=4):
    """Return a single trajectory

    Return a single trajectory of size :math:`(g+1)`-by-:math:`k`
    where :math:`g` is the number of groups,
    and :math:`k` is the number of factors,
    both implied by the dimensions of `group_membership`

    Arguments
    ---------
    group_membership : np.ndarray
        a k-by-g matrix which notes factor membership of groups
    num_levels : int, default=4
        The number of levels in the grid

    Returns
    -------
    np.ndarray
    """

    delta = compute_delta(num_levels)

    # Infer number of groups `g` and number of params `k` from
    # `group_membership` matrix
    num_params = group_membership.shape[0]
    num_groups = group_membership.shape[1]

    # Matrix B - size (g + 1) * g -  lower triangular matrix
    B = np.tril(np.ones([num_groups + 1, num_groups],
                        dtype=int), -1)

    P_star = generate_p_star(num_groups)

    # Matrix J - a (g+1)-by-num_params matrix of ones
    J = np.ones((num_groups + 1, num_params))

    # Matrix D* - num_params-by-num_params matrix which decribes whether
    # factors move up or down
    D_star = np.diag(rd.choice([-1, 1], num_params))

    x_star = generate_x_star(num_params, num_levels)

    # Matrix B* - size (num_groups + 1) * num_params
    B_star = compute_b_star(J, x_star, delta, B,
                            group_membership, P_star, D_star)

    return B_star


def compute_b_star(J, x_star, delta, B, G, P_star, D_star):
    """
    """
    element_a = J[0, :] * x_star
    element_b = np.matmul(G, P_star).T
    element_c = np.matmul(2.0 * B, element_b)
    element_d = np.matmul((element_c - J), D_star)

    b_star = element_a + (delta / 2.0) * (element_d + J)
    return b_star


def generate_p_star(num_groups: int):
    """Describe the order in which groups move

    Arguments
    ---------
    num_groups : int

    Returns
    -------
    np.ndarray
        Matrix P* - size (g-by-g)
    """
    p_star = np.eye(num_groups, num_groups)
    rd.shuffle(p_star)
    return p_star


def generate_x_star(num_params, num_levels):
    """Generate an 1-by-num_params array to represent initial position for EE

    This should be a randomly generated array in the p level grid
    :math:`\omega`

    Arguments
    ---------
    num_params : int
        The number of parameters (factors)
    num_levels : int
        The number of levels

    Returns
    -------
    numpy.ndarray
        The initial starting positions of the trajectory
    """
    x_star = np.zeros((1, num_params))
    delta = compute_delta(num_levels)
    bound = 1 - delta
    grid = np.linspace(0, bound, int(num_levels / 2))

    x_star[0, :] = rd.choice(grid, num_params)

    return x_star


def compute_delta(num_levels):
    """Computes the delta value from number of levels

    Arguments
    ---------
    num_levels : int
        The number of levels

    Returns
    -------
    float
    """
    return num_levels / (2.0 * (num_levels - 1))


def _compute_optimised_trajectories(problem, input_sample, N, k_choices,
                                    local_optimization=False):
    '''
    Calls the procedure to compute the optimum k_choices of trajectories
    from the input_sample.
    If there are groups, then this procedure allocates the groups to the
    correct call here.

    Arguments
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
    '''
    if _has_gurobi is False \
            and local_optimization is False \
            and k_choices > 10:
        msg = "Running optimal trajectories greater than values of 10 \
                will take a long time."
        raise ValueError(msg)

    num_params = problem['num_vars']

    if np.any((input_sample < 0) | (input_sample > 1)):
        raise ValueError("Input sample must be scaled between 0 and 1")

    if _has_gurobi and local_optimization is False:
        # Use global optimization method
        strategy = GlobalOptimisation()
    elif local_optimization:
        # Use local method
        strategy = LocalOptimisation()
    else:
        # Use brute force approach
        strategy = BruteForce()

    if problem.get('groups'):
        num_groups = len(set(problem['groups']))
    else:
        num_groups = None

    context = SampleMorris(strategy)
    output = context.sample(input_sample, N, num_params,
                            k_choices, num_groups)

    return output


def cli_parse(parser):
    parser.add_argument('-l', '--levels', type=int, required=False,
                        default=4, help='Number of grid levels \
                        (Morris only)')
    parser.add_argument('-k', '--k-optimal', type=int, required=False,
                        default=None,
                        help='Number of optimal trajectories \
                        (Morris only)')
    parser.add_argument('-lo', '--local', type=bool, required=True,
                        default=False,
                        help='Use the local optimisation method \
                        (Morris with optimization only)')
    return parser


def cli_action(args):
    rd.seed(args.seed)

    problem = read_param_file(args.paramfile)
    param_values = sample(problem, args.samples, args.levels,
                          args.k_optimal, args.local)

    np.savetxt(args.output, param_values, delimiter=args.delimiter,
               fmt='%.' + str(args.precision) + 'e')


if __name__ == "__main__":
    common_args.run_cli(cli_parse, cli_action)
