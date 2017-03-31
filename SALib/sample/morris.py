from __future__ import division

import numpy as np
import random as rd

from . import common_args
from ..util import scale_samples, read_param_file, compute_groups_matrix
from . optimal_trajectories import return_max_combo

from . morris_util import *
from operator import or_

try:
    from gurobipy import *
except ImportError:
    _has_gurobi = False
else:
    _has_gurobi = True


def sample(problem, N, num_levels, grid_jump, optimal_trajectories=None,
           local_optimization=False):
    """Generates model inputs using the Method of Morris.

    Returns a NumPy matrix containing the model inputs required for Method of
    Morris.  The resulting matrix has :math:`(D+1)*N` rows and :math:`D`
    columns, where :math:`D` is the number of parameters.  These model inputs
    are intended to be used with :func:`SALib.analyze.morris.analyze`.

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
    trajectories, or a local method based which is much faster. While the former
    implements groups, the latter does not.
    Note that the number of factors makes little difference,
    but the ratio between number of optimal trajectories and the sample size
    results in an exponentially increasing number of scores that must be
    computed to find the optimal combination of trajectories.  We suggest going
    no higher than 4 from a pool of 100 samples with the brute force approach.
    With local_optimization = True, it is possible to go higher than the
    previously suggested 4 from 100.

    Parameters
    ----------
    problem : dict
        The problem definition
    N : int
        The number of samples to generate
    num_levels : int
        The number of grid levels
    grid_jump : int
        The grid jump size
    optimal_trajectories : int
        The number of optimal trajectories to sample (between 2 and N)
    local_optimization : bool
        Flag whether to use local optimization according to Ruano et al. (2012)
        Speeds up the process tremendously for bigger N and num_levels.
        Stating this variable to be true causes the function to ignore gurobi.
    """
    if grid_jump >= num_levels:
        raise ValueError("grid_jump must be less than num_levels")

    if problem.get('groups'):
        sample = sample_groups(problem, N, num_levels, grid_jump)
    else:
        sample = sample_oat(problem, N, num_levels, grid_jump)

    if optimal_trajectories:

        assert type(optimal_trajectories) == int, \
            "Number of optimal trajectories should be an integer"

        if optimal_trajectories < 2:
            raise ValueError("The number of optimal trajectories must be set to 2 or more.")
        if optimal_trajectories >= N:
            raise ValueError("The number of optimal trajectories should be less than the number of samples.")

        if _has_gurobi == False and local_optimization == False and optimal_trajectories > 10:
            raise ValueError("Running optimal trajectories greater than values of 10 will take a long time.")

        sample = compute_optimised_trajectories(problem,
                                                sample,
                                                N,
                                                optimal_trajectories,
                                                local_optimization)

    scale_samples(sample, problem['bounds'])
    return sample


def sample_oat(problem, N, num_levels, grid_jump):

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
        perm = np.random.permutation(D)
        P = np.zeros([D, D])
        for i in range(D):
            P[i, perm[i]] = 1

        # starting point for this trajectory
        x_base = np.zeros([D + 1, D])
        for i in range(D):
            x_base[:, i] = (
                rd.choice(np.arange(num_levels - grid_jump))) / (num_levels - 1)

        # Indices to be assigned to X, corresponding to this trajectory
        index_list = np.arange(D + 1) + j * (D + 1)
        delta_diag = np.diag([delta for _ in range(D)])

        X[index_list, :] = 0.5 * \
            (np.mat(B) * np.mat(P) * np.mat(DM) + 1) * \
            np.mat(delta_diag) + np.mat(x_base)

    return X

def sample_groups(problem, N, num_levels, grid_jump):
    '''
    Returns an N(g+1)-by-k array of N trajectories;
    where g is the number of groups and k is the number of factors
    '''
    G, group_names = compute_groups_matrix(problem['groups'], problem['num_vars'])

    if G is None:
        raise ValueError("Please define the matrix G.")
    if type(G) is not np.matrixlib.defmatrix.matrix:
        raise TypeError("Matrix G should be formatted as a numpy matrix")

    k = G.shape[0]
    g = G.shape[1]
    sample = np.zeros((N * (g + 1), k))
    sample = np.array([generate_trajectory(G, num_levels, grid_jump) for n in range(N)])
    return sample.reshape((N * (g + 1), k))


def compute_optimised_trajectories(problem, input_sample, N, k_choices, local_optimization = False):
    '''
    Calls the procedure to compute the optimum k_choices of trajectories
    from the input_sample.
    If there are groups, then this procedure allocates the groups to the
    correct call here.
    '''
    num_params = problem['num_vars']
    groups = compute_groups_matrix(problem['groups'], num_params)

    if np.any((input_sample < 0) | (input_sample > 1)):
        raise ValueError("Input sample must be scaled between 0 and 1")

    if _has_gurobi == True and local_optimization == False:
        maximum_combo = return_max_combo(input_sample,
                                         N,
                                         num_params,
                                         k_choices,
                                         groups)

    else:
        maximum_combo = find_optimum_combination(input_sample,
                                                 N,
                                                 num_params,
                                                 k_choices,
                                                 groups,
                                                 local_optimization)

    num_groups = None
    if groups is not None:
        num_groups = groups[0].shape[1]

    output = compile_output(input_sample,
                            N,
                            num_params,
                            maximum_combo,
                            num_groups)

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
                        default=None, help='Number of optimal trajectories (Morris only)')
    parser.add_argument('-o', '--local', type=bool, required=True,
                        default=False, help='Use the local optimisation method (Morris with optimization only)')
    args = parser.parse_args()

    np.random.seed(args.seed)
    rd.seed(args.seed)

    problem = read_param_file(args.paramfile)
    param_values = sample(problem, args.samples, args.levels, \
                    args.grid_jump, args.k_optimal, args.local)

    np.savetxt(args.output, param_values, delimiter=args.delimiter,
               fmt='%.' + str(args.precision) + 'e')
