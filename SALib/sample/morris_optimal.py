from __future__ import division
import numpy as np
import random as rd
from itertools import combinations, islice
from ..util import read_param_file
from . import common_args
from scipy.spatial.distance import cdist
from scipy.misc import comb as nchoosek
import sys


def compute_distance(m, l, num_params):
    '''
    Computes the distance between two trajectories
    '''

    if np.shape(m) != np.shape(l):
        raise ValueError("Input matrices are different sizes")

    distance = np.array(np.sum(cdist(m, l)), dtype=np.float32)

    return distance


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

    # Now evaluate the (N choose k_choices) possible combinations
    if nchoosek(N, k_choices) >= sys.maxsize:
        raise ValueError("Number of combinations is too large")
    number_of_combinations = int(nchoosek(N, k_choices))

    # First compute the distance matrix for each possible pairing
    # of trajectories and store in a shared-memory array
    distance_matrix = compute_distance_matrix(input_sample,
                                              N,
                                              num_params)


    # Initialise the output array

    chunk = int(1e6)
    if chunk > number_of_combinations:
        chunk = number_of_combinations

    counter = 0
    # Generate a list of all the possible combinations
    #combos = np.array([x for x in combinations(range(N),k_choices)])
    combo_gen = combinations(list(range(N)),k_choices)
    scores = np.empty(number_of_combinations,dtype=np.float32)
    # Generate the pairwise indices once
    pairwise = np.array([y for y in combinations(list(range(k_choices)),2)])

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


def find_maximum(scores, N, k_choices):

    if type(scores) != np.ndarray:
        raise TypeError("Scores input is not a numpy array")

    index_of_maximum = int(scores.argmax())
    maximum_combo = nth(combinations(list(range(N)), k_choices), index_of_maximum, None)
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

    if type(n) != int:
        raise TypeError("n is not an integer")

    return next(islice(iterable, n, None), default)


def make_index_list(N, num_params):
    index_list = []
    for j in range(N):
        index_list.append(np.arange(num_params + 1) + j * (num_params + 1))
    return index_list


def compile_output(input_sample, N, num_params, k_choices, maximum_combo):

    index_list = make_index_list(N, num_params)

    output = np.zeros((len(maximum_combo) * (num_params + 1), num_params))

    for counter, x in enumerate(maximum_combo):
        output[index_list[counter]] = np.array(input_sample[index_list[x]])
    return output


def find_optimum_trajectories(input_sample, N, num_params, k_choices):


    if np.any((input_sample < 0) | (input_sample > 1)):
        raise ValueError("Input sample must be scaled between 0 and 1")

    maximum_combo = find_optimum_combination(input_sample, N, num_params, k_choices)

    return compile_output(input_sample, N, num_params, k_choices, maximum_combo)


def find_optimum_combination(input_sample, N, num_params, k_choices):


    scores = find_most_distant(input_sample,
                               N,
                               num_params,
                               k_choices)

    maximum_combo = find_maximum(scores, N, k_choices)

    return maximum_combo


if __name__ == "__main__":


    parser = common_args.create()
    parser.add_argument('--num-levels', type=int, required=False, default=4, help='Number of grid levels (Morris only)')
    parser.add_argument('--grid-jump', type=int, required=False, default=2, help='Grid jump size (Morris only)')
    parser.add_argument('--k-choices', type=int, required=False, default=4, help='Number of trajectories (optimal trajectories)')
    parser.add_argument('-X', '--input-file', type=str, required=True, default=None, help='Model input file')

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

    X = np.loadtxt(args.input_file, delimiter=args.delim, ndmin=2)

    model_data = find_optimum_trajectories(X, N, num_params, k_choices)

    np.savetxt(args.output, model_data, delimiter=args.delimiter, fmt='%.' + str(args.precision) + 'e')
