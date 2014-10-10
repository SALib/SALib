from __future__ import division
import numpy as np
import random as rd
from itertools import combinations
from util import read_param_file
from sample import common_args
#set up parallel processes, start ipcluster from command line prior!
from IPython.parallel import Client
from esme import morris_sample, num_combinations, \
                 compute_distance_matrix, nth, grouper

import sys
sys.path[0] = '/Users/will2/repository/SALib/SALib/'

import mylib


def pl_find_most_distant(input_sample, N, num_params, k_choices):
    '''
    Finds the 'k_choices' most distant choices from the
    'N' trajectories contained in 'input_sample'
    '''
    # First compute the distance matrix for each possible pairing
    # of trajectories and store in a shared-memory array

    # Now evaluate the (N choose k_choices) possible combinations
    number_of_combinations = num_combinations(N, k_choices)
    # Initialise the output array

    chunk = int(1e6)
    if chunk > number_of_combinations:
        chunk = int(number_of_combinations)

    # Generate a list of all the possible combinations
    #combos = np.array([x for x in combinations(range(N),k_choices)])
    combo_gen = combinations(range(N),k_choices)
    scores = np.empty(number_of_combinations,dtype=np.float16)

    rc=Client()
    dview=rc[:]

    #...do stuff to get iterable_of_a and b,c,d....

    distance_matrix = compute_distance_matrix(input_sample,
                                              N,
                                              num_params)

    function = lambda combos: mylib.pl_mappable(combos, k_choices, distance_matrix)

    mydict=dict(k_choices=k_choices,
                distance_matrix=distance_matrix)
    dview.push(mydict)
    dview.sync_imports('import numpy as np')
    dview.execute('import mylib')
    scores = dview.map_sync(function, [x for x in grouper(chunk, combo_gen)])

    return np.ravel(scores)


def pl_mappable2(combos, k_choices, distance_matrix):
    pairwise = np.array([y for y in combinations(range(k_choices),2)])
    combos = np.array(combos)
    # Create a list of all pairwise combination for each combo in combos
    combo_list = combos[:,pairwise[:,]]
    all_distances = distance_matrix[[combo_list[:,:,1], combo_list[:,:,0]]]
    new_scores = np.sqrt(np.einsum('ij,ij->i', all_distances, all_distances))
    return new_scores


def find_index_of_maximum(scores):

    if type(scores) != np.ndarray:
        raise TypeError("Scores input is not a numpy array")
    return scores.argmax()


if __name__ == "__main__":

    parser = common_args.create()
    parser.add_argument('--num-levels', type=int, required=False, default=4, help='Number of grid levels (Morris only)')
    parser.add_argument('--grid-jump', type=int, required=False, default=2, help='Grid jump size (Morris only)')
    parser.add_argument('--k-choices', type=int, required=False, default=4, help='Number of choices (optimised trajectory)')

    args = parser.parse_args()

    np.random.seed(args.seed)
    rd.seed(args.seed)


    rc=Client()
    dview=rc[:]


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


    number_of_combinations = num_combinations(N, k_choices)

    chunk = int(1e6)
    if chunk > number_of_combinations:
        chunk = int(number_of_combinations)
    counter = 0

    # Generate a list of all the possible combinations
    combo_gen = combinations(range(N),k_choices)
    scores = np.zeros(number_of_combinations,dtype=np.float32)

    distance_matrix = compute_distance_matrix(input_data,
                                              N,
                                              num_params)

    function = lambda x: mylib.pl_mappable(x, k_choices, distance_matrix)

    mydict=dict(k_choices=k_choices,
                distance_matrix=distance_matrix)

    dview.push(mydict)

    dview.sync_imports('import sys')

    dview.execute('sys.path[0] = ''/Users/will2/repository/SALib/SALib''')
    dview.execute('import mylib')

    for combos in grouper(chunk, combo_gen):
        chunkette = len(combos)
        scores[(counter*chunk):((counter*chunk)+chunkette)] = dview.map_sync(function, combos)
        counter += 1

    index_of_maximum = np.argmax(scores, 0)

    index_list = []
    for j in range(N):
        index_list.append(np.arange(num_params + 1) + j * (num_params + 1))

    maximum_combo = nth(combinations(range(N), k_choices), index_of_maximum)

    output = np.zeros((np.size(maximum_combo) * (num_params + 1), num_params))
    for counter, x in enumerate(maximum_combo):
        output[index_list[counter]] = np.array(input_data[index_list[x]])

    np.savetxt(args.output, output, delimiter=args.delimiter, fmt='%.' + str(args.precision) + 'e')
