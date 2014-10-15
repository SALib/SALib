from __future__ import division
import numpy as np
import random as rd
from math import factorial
from itertools import combinations, islice
from util import scale_samples, read_param_file
from sample import common_args
from scipy.spatial.distance import cdist


def morris_sample(N, num_params, bounds, num_levels, grid_jump, seed=19284982948):
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

    np.random.seed(seed)
    rd.seed(seed)

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

    distance = np.array(np.sum(cdist(m, l)), dtype=np.float32)

    return distance


def num_combinations(n, k):
    numerator = factorial(n)
    denominator = (factorial(k) * factorial(n - k))
    answer = numerator / denominator
    return long(answer)


def compute_distance_matrix(input_sample, N, num_params):
    index_list = []
    distance_matrix = np.zeros((N, N), dtype=np.float32)

    for j in range(N):
        index_list.append(np.arange(num_params + 1) + j * (num_params + 1))

    for j in range(N):
        input_1 = input_sample[index_list[j]]
        for k in range(j + 1, N):
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

    chunk = int(1e6)
    if chunk > number_of_combinations:
        chunk = number_of_combinations

    counter = 0
    # Generate a list of all the possible combinations
    #combos = np.array([x for x in combinations(range(N),k_choices)])
    combo_gen = combinations(range(N),k_choices)
    scores = np.empty(number_of_combinations,dtype=np.float32)
    # Generate the pairwise indices once
    pairwise = np.array([y for y in combinations(range(k_choices),2)])

    for combos in grouper(chunk, combo_gen):
        scores[(counter*chunk):((counter+1)*chunk)] = mappable(combos, pairwise, distance_matrix)
        counter += 1
    return scores


def mappable(combos, pairwise, distance_matrix):
    '''
    Obtains scores from the distance_matrix for each pairwise combination
    held in the combos array
    '''
    import numpy as np
    combos = np.array(combos)
    # Create a list of all pairwise combination for each combo in combos
    combo_list = combos[:,pairwise[:,]]
    all_distances = distance_matrix[[combo_list[:,:,1], combo_list[:,:,0]]]
    new_scores = np.sqrt(np.einsum('ij,ij->i', all_distances, all_distances))
    return new_scores


def pl_mappable(combos):
    import numpy as np
    pairwise = np.array([y for y in combinations(range(k_choices),2)])
    combos = np.array(combos)
    # Create a list of all pairwise combination for each combo in combos
    combo_list = combos[:,pairwise[:,]]
    all_distances = distance_matrix[[combo_list[:,:,1], combo_list[:,:,0]]]
    new_scores = np.sqrt(np.einsum('ij,ij->i', all_distances, all_distances))
    return new_scores


def find_maximum(scores, N, k_choices):

    if type(scores) != np.ndarray:
        raise TypeError("Scores input is not a numpy array")

    index_of_maximum = scores.argmax()
    maximum_combo = nth(combinations(range(N), k_choices), index_of_maximum)
    return maximum_combo


def grouper(n, iterable):
    it = iter(iterable)
    while True:
       chunk = tuple(islice(it, n))
       if not chunk:
           return
       yield chunk


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


def nth(iterable, n, default=None):
    "Returns the nth item or a default value"
    return next(islice(iterable, n, None), default)


def find_optimum_trajectories(input_sample, N, num_params, k_choices):

    scores = find_most_distant(input_sample,
                               N,
                               num_params,
                               k_choices)

    index_of_maximum = scores.argmax()
    index_list = []
    for j in range(N):
        index_list.append(np.arange(num_params + 1) + j * (num_params + 1))

    maximum_combo = nth(combinations(range(N), k_choices), index_of_maximum)
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
    parser.add_argument('--k-choices', type=int, required=False, default=4, help='Number of choices (optimised trajectory)')

    args = parser.parse_args()

    np.random.seed(args.seed)
    rd.seed(args.seed)


    param_file = args.paramfile
    pf = read_param_file(param_file)
    N = args.samples
    num_params = pf['num_vars']
    bounds = pf['bounds']
    k_choices = args.k_choices
    p_levels = int(args.num_levels)
    grid_step = int(args.grid_jump)
    # Generates N(D + 1) x D matrix of samples
    input_data = morris_sample(N,
                        num_params,
                        bounds,
                        num_levels=p_levels,
                        grid_jump=grid_step)
    distance_matrix = compute_distance_matrix(input_data,
                                              N,
                                              num_params)

    model_data = find_optimum_trajectories(input_data, N, num_params, k_choices)
    np.savetxt(args.output, model_data, delimiter=' ')

    #np.savetxt(args.output, model_data, delimiter=args.delimiter, fmt='%.' + str(args.precision) + 'e')
