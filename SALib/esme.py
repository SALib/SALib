from __future__ import division
import numpy as np
import random as rd
from math import factorial
from itertools import combinations
from util import scale_samples, read_param_file
from sample import common_args


def morris_sample(N, num_params, bounds, num_levels, grid_jump):
    '''
    Generates N('num_params' + 1) x 'num_params' matrix of Morris samples (OAT)
    '''

    if num_params < 1:
        raise ValueError("num_params must be greater than or equal to 1")
    if N < 1:
        raise ValueError("N should be greater than 0")
    if type(bounds) != "list":
        TypeError("bounds should be a list")

    D = num_params

    # orientation matrix B: lower triangular (1) + upper triangular (-1)
    B = np.tril(np.ones([D+1, D], dtype=np.int), -1) + np.triu(-1*np.ones([D+1,D], dtype=np.int))

    # grid step delta, and final sample matrix X
    delta = grid_jump / (num_levels - 1)
    X = np.empty([N*(D+1), D], dtype=np.float32)

    # Create N trajectories. Each trajectory contains D+1 parameter sets.
    # (Starts at a base point, and then changes one parameter at a time)
    for j in range(N):

        # directions matrix DM - diagonal matrix of either +1 or -1
        DM = np.diag([rd.choice([-1,1]) for _ in range(D)])

        # permutation matrix P
        perm = np.random.permutation(D)
        P = np.zeros([D,D], dtype=np.float32)
        for i in range(D):
            P[i, perm[i]] = 1

        # starting point for this trajectory
        x_base = np.empty([D+1, D], dtype=np.float32)
        for i in range(D):
            x_base[:,i] = (rd.choice(np.arange(num_levels - grid_jump))) / (num_levels - 1)

        # Indices to be assigned to X, corresponding to this trajectory
        index_list = np.arange(D+1) + j*(D + 1)
        delta_diag = np.diag([delta for _ in range(D)])

        X[index_list,:] = 0.5*(np.mat(B)*np.mat(P)*np.mat(DM) + 1) * np.mat(delta_diag) + np.mat(x_base)

    scale_samples(X, bounds)
    return X


def compute_distance(m, l, num_params):
    '''
    Computes the distance between two trajectories
    '''

    if np.shape(m) != np.shape(l):
        raise ValueError("Input matrices are different sizes")

    output = np.zeros([np.size(m, 0), np.size(m, 0)],
                      dtype=np.float32)

    for i in range(0, num_params+1):
        for j in range(0, num_params+1):
            output[i, j] = np.sum(np.square(np.subtract(m[i, :], l[j, :])))

    distance = np.array(np.sum(np.sqrt(output), (0, 1)),
                        dtype=np.float32)

    return distance


def num_combinations(n, k):
    numerator = factorial(n)
    denominator = (factorial(k) * factorial(n - k))
    answer = numerator / denominator
    return answer


def compute_distance_matrix(input_sample, N, num_params):
    index_list = []
    distance_matrix = np.zeros((N, N), dtype=np.float32)

    for j in range(N):
        index_list.append(np.arange(num_params + 1) + j * (num_params + 1))

    for j in range(N):
        for k in range(j + 1, N):
            input_1 = input_sample[index_list[j]]
            input_2 = input_sample[index_list[k]]
            distance_matrix[k, j] = compute_distance(input_1, input_2,
                                                     num_params)
    return distance_matrix


def find_most_distant(input_sample, N, num_params, k_choices):
    '''
    Finds the 'k_choices' most distant choices from the
    'N' trajectories contained in 'input_sample'
    '''

    # First compute the distance matrix for each possible pairing
    # of trajectories and store in a shared-memory array
    distance_matrix = compute_distance_matrix(input_sample,
                                              N,
                                              num_params)

    # Now evaluate the (N choose k_choices) possible combinations
    number_of_combinations = num_combinations(N, k_choices)
    # Initialise the output array
    output = np.zeros((number_of_combinations),
                      dtype=np.float32)

    # Generate a list of all the possible combinations
    combos = [t for t in combinations(range(N), k_choices)]

    def Map(one_combination):
        '''
        Get a list of the pair-combinations from 'one_combination'

        and return the spread, from the sum of absolute distance
        of each combination
        '''

        list_of_pairs = combinations(one_combination, 2)

        distances = [distance_matrix[x[1], x[0]] for x in list_of_pairs]

        return np.sqrt(np.einsum('i,i', distances, distances))

    output = np.array(map(Map, combos), dtype=np.float32)

    return output, combos


def find_maximum(scores, combinations):

    if type(scores) != np.ndarray:
        raise TypeError("Scores input is not a numpy array")

    index_of_maximum = scores.argmax()
    print scores[index_of_maximum]
    return combinations[index_of_maximum]


def find_optimimum_trajectories(input_sample, N, num_params, k_choices):

    scores, combinations = find_most_distant(input_sample,
                                             N,
                                             num_params,
                                             k_choices)
    index_of_maximum = scores.argmax()
    index_list = []
    for j in range(N):
        index_list.append(np.arange(num_params + 1) + j * (num_params + 1))

    maximum_combo = combinations[index_of_maximum]
    output = np.zeros((np.size(maximum_combo) * (num_params + 1), num_params))
    for counter, x in enumerate(maximum_combo):
        output[index_list[counter]] = np.array(input_sample[index_list[x]])
    return output


class OptimisedTrajectories(object):

    def __init__(self, N, param_file, num_levels, grid_jump):
        self.N = N
        self.param_file = param_file

        pf = read_param_file(param_file)
        self.num_params = pf['num_vars']
        self.bounds = pf['bounds']

        self.num_levels = num_levels
        self.grid_jump = grid_jump

        self.k_choices = k_choices
        self.sample_inputs = self.morris_sample(self.N,
                                         self.num_params,
                                         self.bounds,
                                         self.num_levels,
                                         self.grid_jump)
        self.distance_matrix = \
            np.array(compute_distance_matrix(self.sample_inputs,
                                             self.N,
                                             self.num_params), dtype=np.float32)

        self.optimised_inputs = \
                        self.find_optimimum_trajectories(self.sample_inputs,
                                                        self.N,
                                                        self.num_params,
                                                        self.k_choices)



if __name__ == "__main__":


    parser = common_args.create()
    parser.add_argument('--num-levels', type=int, required=False, default=4, help='Number of grid levels (Morris only)')
    parser.add_argument('--grid-jump', type=int, required=False, default=2, help='Grid jump size (Morris only)')
    args = parser.parse_args()


    np.random.seed(args.seed)
    rd.seed(args.seed)


    param_file = args.paramfile
    pf = read_param_file(param_file)
    N = 6
    num_params = pf['num_vars']
    bounds = pf['bounds']
    print bounds
    k_choices = 4
    p_levels = int(args.num_levels)
    grid_step = int(args.grid_jump)
    # Generates N(D + 1) x D matrix of samples
    input_data = morris_sample(N,
                        num_params,
                        bounds,
                        num_levels=p_levels,
                        grid_jump=grid_step)
    print input_data
    model_data = find_optimimum_trajectories(input_data, N, num_params, k_choices)
    print model_data
    np.savetxt(args.output, model_data, delimiter=' ')

    #np.savetxt(args.output, model_data, delimiter=args.delimiter, fmt='%.' + str(args.precision) + 'e')
