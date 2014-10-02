from SALib.sample import morris_oat
import numpy as np
from math import factorial
from itertools import combinations
from SALib.util import read_param_file


def compute_distance(m, l):
    '''
    Computes the distance between two trajectories
    '''

    if np.shape(m) != np.shape(l):
        raise ValueError("Input matrices are different sizes")

    output = np.zeros([np.size(m, 0), np.size(m, 0), np.size(m, 1)],
                      dtype=np.float16)

    for i in range(0, 3):
        for j in range(0, 3):
            for z in range(0, 2):
                output[i, j, z] = np.square(m[i, z] - l[j, z])

    distance = np.array(np.sum(np.sqrt(np.sum(output, 2)), (0, 1)),
                        dtype=np.float16)

    return distance


def num_combinations(n, k):
    numerator = factorial(n)
    denominator = (factorial(k) * factorial(n - k))
    answer = numerator / denominator
    return answer


def compute_distance_matrix(input_sample, N, num_params):
    index_list = []
    distance_matrix = np.zeros((N, N), dtype=np.float16)

    for j in range(N):
        index_list.append(np.arange(num_params + 1) + j * (num_params + 1))

    for j in range(N):
        for k in range(j + 1, N):
            input_1 = input_sample[index_list[j]]
            input_2 = input_sample[index_list[k]]
            distance_matrix[k, j] = compute_distance(input_1, input_2)
    return distance_matrix


def find_most_distant(input_sample, N, num_params, k_choices):
    '''
    Finds the 'k_choices' most distant choices from the
    'N' trajectories contained in 'input_sample'
    '''

    # First compute the distance matrix for each possible pairing
    # of trajectories
    distance_matrix = np.array(compute_distance_matrix(input_sample,
                                                       N,
                                                       num_params),
                               dtype=np.float16)

    # Now iterate through each possible combination to (N choose k_choices)
    number_of_combinations = num_combinations(N, k_choices)
    number_of_pairings = num_combinations(k_choices, 2)

    output = np.zeros((number_of_combinations, number_of_pairings),
                      dtype=np.float16)

    # Generate a list of all the possible combinationations
    combos = [t for t in combinations(range(N), k_choices)]

    for counter, t in enumerate(combos):
        for counter2, k in enumerate(combinations(t, 2)):
            output[counter, counter2] = np.square(distance_matrix[k[1], k[0]])
    scores = np.sqrt(np.sum(output, 1))
    index_of_maximum = scores.argmax()
    print scores[index_of_maximum]
    return combos[index_of_maximum]


if __name__ == "__main__":
    param_file = 'esme_param.txt'
    pf = read_param_file(param_file)
    N = 100
    num_params = pf['num_vars']
    k_choices = 4
    p_levels = 4
    grid_step = 2
    # Generates N(D + 1) x D matrix of samples
    input_data = morris_oat.sample(N,
                                   param_file,
                                   num_levels=p_levels,
                                   grid_jump=grid_step)
    print find_most_distant(input_data, N, num_params, k_choices)
