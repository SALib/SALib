from __future__ import division
import numpy as np
import random as rd
from . import common_args
from . morris_util import *
from ..util import scale_samples, read_param_file

'''
Three variants of Morris' sampling for
elementary effects:
        - vanilla Morris
        - optimised trajectories (Campolongo's enhancements from 2007)
        - groups with optimised trajectories (again Campolongo 2007)

At present, optimised trajectories is implemented using a brute-force
approach, which can be very slow, especially if you require more than four
trajectories.  Note that the number of factors makes little difference,
but the ratio between number of optimal trajectories and the sample size
results in an exponentially increasing number of scores that must be
computed to find the optimal combination of trajectories.

I suggest going no higher than 4 from a pool of 100 samples.

Suggested enhancements:
    - a parallel brute force method (incomplete)
    - a combinatorial optimisation approach (completed, but dependencies are
      not open-source)
'''


def sample(problem, N, num_levels, grid_jump, optimal_trajectories=None):

    if grid_jump >= num_levels:
        raise ValueError("grid_jump must be less than num_levels")

    if problem['groups'] is None:
        sample = sample_oat(problem, N, num_levels, grid_jump)
    else:
        sample = sample_groups(problem, N, num_levels, grid_jump)

    if optimal_trajectories is not None:
        if optimal_trajectories >= N:
            raise ValueError("The number of optimal trajectories should be less than the number of samples.")
        elif optimal_trajectories > 10:
            raise ValueError("Running optimal trajectories greater than values of 10 will take a long time.")
        elif optimal_trajectories < 2:
            raise ValueError("The number of optimal trajectories must be set to 2 or more.")
        else:
            sample = find_optimum_trajectories(problem, sample, N, optimal_trajectories)

    scale_samples(sample, problem['bounds'])
    return sample


def sample_oat(problem, N, num_levels, grid_jump):

    D = problem['num_vars']

    # orientation matrix B: lower triangular (1) + upper triangular (-1)
    B = np.tril(np.ones([D + 1, D], dtype=int), -1) + \
        np.triu(-1 * np.ones([D + 1, D], dtype=int))

    # grid step delta, and final sample matrix X
    delta = grid_jump / (num_levels - 1)
    X = np.empty([N * (D + 1), D])

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
        x_base = np.empty([D + 1, D])
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
    G, group_names = problem['groups']

    if G is None:
        raise ValueError("Please define the matrix G.")
    if type(G) is not np.matrixlib.defmatrix.matrix:
        raise TypeError("Matrix G should be formatted as a numpy matrix")

    k = G.shape[0]
    g = G.shape[1]
    sample = np.empty((N*(g + 1), k))
    sample = np.array([generate_trajectory(G, num_levels, grid_jump) for n in range(N)])
    return sample.reshape((N*(g + 1), k))


def find_optimum_trajectories(problem, input_sample, N, k_choices):
    '''
    Calls the procedure to compute the optimum k_choices of trajectories
    from the input_sample.
    If there are groups, then this procedure allocates the groups to the
    correct call here.
    '''
    num_params = problem['num_vars']
    groups = problem['groups']
    
    if np.any((input_sample < 0) | (input_sample > 1)):
        raise ValueError("Input sample must be scaled between 0 and 1")
    
    maximum_combo = find_optimum_combination(input_sample, 
                                             N, 
                                             num_params, 
                                             k_choices, 
                                             groups)

    num_groups = None
    if groups != None:
        num_groups = groups[0].shape[1]

    return compile_output(input_sample, 
                          N, 
                          num_params, 
                          maximum_combo, 
                          num_groups)


if __name__ == "__main__":

    parser = common_args.create()
    parser.add_argument('-l','--levels', type=int, required=False,
                        default=4, help='Number of grid levels (Morris only)')
    parser.add_argument('--grid-jump', type=int, required=False,
                        default=2, help='Grid jump size (Morris only)')
    parser.add_argument('-k','--k-optimal', type=int, required=False,
                        default=None, help='Number of optimal trajectories (Morris only)')
    args = parser.parse_args()

    np.random.seed(args.seed)
    rd.seed(args.seed)

    problem = read_param_file(args.paramfile)
    param_values = sample(problem, args.samples, args.levels, \
                    args.grid_jump, args.k_optimal)

    np.savetxt(args.output, param_values, delimiter=args.delimiter,
               fmt='%.' + str(args.precision) + 'e')
