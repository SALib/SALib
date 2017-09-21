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

import numpy as np

import numpy.random as rd

from . gurobi import GlobalOptimisation
from . local import LocalOptimisation
from . brute import BruteForce

from . strategy import SampleMorris

from SALib.sample import common_args
from SALib.util import scale_samples, read_param_file, compute_groups_matrix

try:
    import gurobipy
except ImportError:
    _has_gurobi = False
else:
    _has_gurobi = True


def sample(problem, N, num_levels, grid_jump, optimal_trajectories=None,
           local_optimization=True):
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
    num_levels : int
        The number of grid levels
    grid_jump : int
        The grid jump size
    optimal_trajectories : int
        The number of optimal trajectories to sample (between 2 and N)
    local_optimization : bool, default=True
        Flag whether to use local optimization according to Ruano et al. (2012)
        Speeds up the process tremendously for bigger N and num_levels.
        If set to ``False`` brute force method is used, unless ``gurobipy`` is
        available

    Returns
    -------
    sample : numpy.ndarray
        Returns a numpy.ndarray containing the model inputs required for Method
        of Morris. The resulting matrix has :math:`(G/D+1)*N/T` rows and
        :math:`D` columns, where :math:`D` is the number of parameters.
    """
    if grid_jump >= num_levels:
        raise ValueError("grid_jump must be less than num_levels")

    if problem.get('groups'):
        sample = _sample_groups(problem, N, num_levels, grid_jump)
    else:
        sample = _sample_oat(problem, N, num_levels, grid_jump)

    if optimal_trajectories:

        sample = _compute_optimised_trajectories(problem,
                                                 sample,
                                                 N,
                                                 optimal_trajectories,
                                                 local_optimization)

    scale_samples(sample, problem['bounds'])
    return sample


def _sample_oat(problem, N, num_levels, grid_jump):
    """Generate trajectories without groups

    Arguments
    ---------
    problem : dict
        The problem definition
    N : int
        The number of samples to generate
    num_levels : int
        The number of grid levels
    grid_jump : int
        The grid jump size
    """

    D = problem['num_vars']

    # orientation matrix B: lower triangular (1) + upper triangular (-1)
    B = np.tril(np.ones([D + 1, D], dtype=int), -1) + \
        np.triu(-1 * np.ones([D + 1, D], dtype=int))

    # grid step delta, and final sample matrix X
    delta = grid_jump / (num_levels - 1)
    X = np.zeros([N * (D + 1), D])

    # Create N trajectories. Each trajectory contains D+1 parameter sets.
    # (Starts at a base point, and then changes one parameter at a time)
    for j in range(N):

        # directions matrix DM - diagonal matrix of either +1 or -1
        DM = np.diag([rd.choice([-1, 1]) for _ in range(D)])

        # permutation matrix P
        perm = rd.permutation(D)
        P = np.zeros([D, D])
        for i in range(D):
            P[i, perm[i]] = 1

        # starting point for this trajectory
        x_base = np.zeros([D + 1, D])
        for i in range(D):
            x_base[:, i] = (
                rd.choice(np.arange(num_levels - grid_jump))) \
                / (num_levels - 1)

        # Indices to be assigned to X, corresponding to this trajectory
        index_list = np.arange(D + 1) + j * (D + 1)
        delta_diag = np.diag([delta for _ in range(D)])

        X[index_list, :] = 0.5 * \
            (np.mat(B) * np.mat(P) * np.mat(DM) + 1) * \
            np.mat(delta_diag) + np.mat(x_base)

    return X


def _sample_groups(problem, N, num_levels, grid_jump):
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
    num_levels : int
        The number of grid levels
    grid_jump : int
        The grid jump size

    Returns
    -------
    numpy.ndarray
    """
    group_membership, _ = compute_groups_matrix(problem['groups'])

    if group_membership is None:
        raise ValueError("Please define the matrix group_membership.")
    if not isinstance(group_membership, np.matrixlib.defmatrix.matrix):
        raise TypeError("Matrix group_membership should be formatted \
                         as a numpy matrix")

    num_params = group_membership.shape[0]
    num_groups = group_membership.shape[1]
    sample = np.zeros((N * (num_groups + 1), num_params))
    sample = np.array([generate_trajectory(group_membership,
                                           num_levels,
                                           grid_jump)
                       for n in range(N)])
    return sample.reshape((N * (num_groups + 1), num_params))


def generate_trajectory(group_membership, num_levels, grid_jump):
    """Return a single trajectory

    Return a single trajectory of size :math:`(g+1)`-by-:math:`k`
    where :math:`g` is the number of groups,
    and :math:`k` is the number of factors,
    both implied by the dimensions of `group_membership`

    Arguments
    ---------
    group_membership : np.ndarray
        a k-by-g matrix which notes factor membership of groups
    num_levels : int
        integer describing number of levels
    grid_jump : int
        recommended to be equal to :math:`p / (2(p-1))`
        where :math:`p` is `num_levels`

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
    B = np.matrix(np.tril(np.ones([num_groups + 1, num_groups],
                                  dtype=int), -1))

    P_star = np.asmatrix(generate_p_star(num_groups))

    # Matrix J - a (g+1)-by-num_params matrix of ones
    J = np.matrix(np.ones((num_groups + 1, num_params)))

    # Matrix D* - num_params-by-num_params matrix which decribes whether
    # factors move up or down
    D_star = np.diag([rd.choice([-1, 1]) for _ in range(num_params)])

    x_star = np.asmatrix(generate_x_star(num_params, num_levels, grid_jump))

    # Matrix B* - size (num_groups + 1) * num_params
    B_star = compute_b_star(J, x_star, delta, B,
                            group_membership, P_star, D_star)

    return B_star


def compute_b_star(J, x_star, delta, B, G, P_star, D_star):
    """
    """
    b_star = J[:, 0] * x_star + \
        (delta / 2) * ((2 * B * (G * P_star).T - J)
                       * D_star + J)
    return b_star


def generate_p_star(num_groups):
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


def generate_x_star(num_params, num_levels, grid_step):
    """Generate an 1-by-num_params array to represent initial position for EE

    This should be a randomly generated array in the p level grid
    :math:`\omega`

    Arguments
    ---------
    num_params : int
        The number of parameters (factors)
    num_levels : int
        The number of levels
    grid_step : int
        The grid step used

    """
    x_star = np.zeros(num_params)

    delta = compute_delta(num_levels)
    bound = 1 - delta
    grid = np.linspace(0, bound, grid_step)

    for i in range(num_params):
        x_star[i] = rd.choice(grid)
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
    return float(num_levels) / (2 * (num_levels - 1))


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


if __name__ == "__main__":

    parser = common_args.create()

    parser.add_argument(
        '-n', '--samples', type=int, required=True, help='Number of Samples')
    parser.add_argument('-l', '--levels', type=int, required=False,
                        default=4, help='Number of grid levels (Morris only)')
    parser.add_argument('--grid-jump', type=int, required=False,
                        default=2, help='Grid jump size (Morris only)')
    parser.add_argument('-k', '--k-optimal', type=int, required=False,
                        default=None,
                        help='Number of optimal trajectories (Morris only)')
    parser.add_argument('-o', '--local', type=bool, required=True,
                        default=False,
                        help='Use the local optimisation method \
                              (Morris with optimization only)')
    args = parser.parse_args()

    rd.seed(args.seed)

    problem = read_param_file(args.paramfile)
    param_values = sample(problem, args.samples, args.levels,
                          args.grid_jump, args.k_optimal, args.local)

    np.savetxt(args.output, param_values, delimiter=args.delimiter,
               fmt='%.' + str(args.precision) + 'e')
