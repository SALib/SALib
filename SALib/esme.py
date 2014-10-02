import numpy as np
import random as rd
from math import factorial
from itertools import combinations
from util import scale_samples, read_param_file


# Generate N(D + 1) x D matrix of Morris samples (OAT)
def sample(N, param_file, num_levels, grid_jump):

    pf = read_param_file(param_file)
    D = pf['num_vars']

    # orientation matrix B: lower triangular (1) + upper triangular (-1)
    B = np.tril(np.ones([D+1, D], dtype=int), -1) + np.triu(-1*np.ones([D+1,D], dtype=int))

    # grid step delta, and final sample matrix X
    delta = grid_jump / (num_levels - 1)
    X = np.empty([N*(D+1), D])

    # Create N trajectories. Each trajectory contains D+1 parameter sets.
    # (Starts at a base point, and then changes one parameter at a time)
    for j in range(N):

        # directions matrix DM - diagonal matrix of either +1 or -1
        DM = np.diag([rd.choice([-1,1]) for _ in range(D)])

        # permutation matrix P
        perm = np.random.permutation(D)
        P = np.zeros([D,D])
        for i in range(D):
            P[i, perm[i]] = 1

        # starting point for this trajectory
        x_base = np.empty([D+1, D])
        for i in range(D):
            x_base[:,i] = (rd.choice(np.arange(num_levels - grid_jump))) / (num_levels - 1)

        # Indices to be assigned to X, corresponding to this trajectory
        index_list = np.arange(D+1) + j*(D + 1)
        delta_diag = np.diag([delta for _ in range(D)])

        X[index_list,:] = 0.5*(np.mat(B)*np.mat(P)*np.mat(DM) + 1) * np.mat(delta_diag) + np.mat(x_base)

    scale_samples(X, pf['bounds'])
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
            distance_matrix[k, j] = compute_distance(input_1, input_2, num_params)
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
                               dtype=np.float32)

    # Now iterate through each possible combination to (N choose k_choices)
    number_of_combinations = num_combinations(N, k_choices)

    output = np.zeros((number_of_combinations),
                      dtype=np.float32)

    # Generate a list of all the possible combinations
    combos = [t for t in combinations(range(N), k_choices)]

    for counter, t in enumerate(combos):
        output[counter] = reduce(add,
                                 map(lambda x:
                                        np.square(distance_matrix[x[1], x[0]]),
                                     combinations(t, 2))
                                )
    scores = np.sqrt(output)
    return scores, combos


def add(x,y):
    return x + y


def find_maximum(scores, combinations):
    index_of_maximum = scores.argmax()
    print scores[index_of_maximum]
    return combinations[index_of_maximum]


if __name__ == "__main__":
    param_file = 'esme_param.txt'
    pf = read_param_file(param_file)
    N = 100
    num_params = pf['num_vars']
    k_choices = 10
    p_levels = 4
    grid_step = 2
    # Generates N(D + 1) x D matrix of samples
    input_data = sample(N,
                                   param_file,
                                   num_levels=p_levels,
                                   grid_jump=grid_step)
    scores, combos = find_most_distant(input_data, N, num_params, k_choices)
    print find_maximum(scores, combos)
