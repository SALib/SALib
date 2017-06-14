'''
Helper functions for Morris trajectories
(Generating group samples, and optimizing trajectory distances)
'''

from __future__ import division

from itertools import combinations, islice
import sys

from scipy.misc import comb as nchoosek
from scipy.spatial.distance import cdist

import numpy as np
import random as rd


def generate_trajectory(G, num_levels, grid_jump):
    """Return a single trajectory

    Return a single trajectory of size :math:`(g+1)`-by-:math:`k`
    where :math`g` is the number of groups,
    and :math:`k` is the number of factors,
    both implied by the dimensions of G

    Arguments
    ---------
    G : np.ndarray
        a k-by-g matrix which notes factor membership of groups
    num_levels : int
        integer describing number of levels
    grid_jump : int
        recommended to be equal to :math:`p / (2(p-1))`
        where :math:`p` is num_levels

    Returns
    -------
    np.ndarray
    """

    delta = compute_delta(num_levels)

    # Infer number of groups g and number of params k from matrix G
    k = G.shape[0]
    g = G.shape[1]

    # Matrix B - size (g + 1) * g -  lower triangular matrix
    B = np.matrix(np.tril(np.ones([g + 1, g], dtype=int), -1))

    P_star = np.asmatrix(generate_P_star(g))

    # Matrix J - a (g+1)-by-k matrix of ones
    J = np.matrix(np.ones((g + 1, k)))

    # Matrix D* - k-by-k matrix which decribes whether factors move up or down
    D_star = np.diag([rd.choice([-1, 1]) for _ in range(k)])

    x_star = np.asmatrix(generate_x_star(k, num_levels, grid_jump))

    # Matrix B* - size (g + 1) * k
    B_star = compute_B_star(J, x_star, delta, B, G, P_star, D_star)

    return B_star


def compute_B_star(J, x_star, delta, B, G, P_star, D_star):
    """
    """
    B_star = J[:, 0] * x_star + \
        (delta / 2) * ((2 * B * (G * P_star).T - J)
                       * D_star + J)
    return B_star


def generate_P_star(g):
    """Describe the order in which groups move

    Arguments
    ---------
    g : int

    Returns
    -------
    np.ndarray
        Matrix P* - size (g-by-g)
    """
    P_star = np.eye(g, g)
    np.random.shuffle(P_star)
    return P_star


def generate_x_star(k, num_levels, grid_step):
    '''
    Generate an 1-by-k array to represent initial position for EE
    This should be a randomly generated array in the p level grid :math:\omega
    '''
    x_star = np.zeros(k)

    delta = compute_delta(num_levels)
    bound = 1 - delta
    grid = np.linspace(0, bound, grid_step)

    for i in range(k):
        x_star[i] = rd.choice(grid)
    return x_star


def compute_delta(num_levels):
    return float(num_levels) / (2 * (num_levels - 1))


def check_input_sample(input_sample, num_params, N):
    '''
    Checks input sample is:
        - the correct size
        - values between 0 and 1
    '''
    assert type(input_sample) == np.ndarray, \
        "Input sample is not an numpy array"
    assert input_sample.shape[0] == (num_params + 1) * N, \
        "Input sample does not match number of parameters or groups"
    assert np.any((input_sample >= 0) | (input_sample <= 1)), \
        "Input sample must be scaled between 0 and 1"


def compute_distance(m, l):
    '''
    Computes the distance between two trajectories
    '''

    if np.shape(m) != np.shape(l):
        raise ValueError("Input matrices are different sizes")
    if np.array_equal(m, l):
        print("Trajectory %s and %s are equal" % (m, l))
        distance = 0
    else:
        distance = np.array(np.sum(cdist(m, l)), dtype=np.float32)

    return distance


def compute_distance_matrix(input_sample, N, num_params, groups=None,
                            local_optimization=False):
    """Computes the distance between every trajectory

    Arguments
    ---------
    input_sample : numpy.ndarray
        The input sample of trajectories for which to compute
        the distance matrix
    N : int
        The number of samples
    num_params : int
        The number of factors
    groups : tuple, default=None
        the mapping between factors and groups
    local_optimization : bool, default=False

    Returns
    -------
    distance_matrix : numpy.ndarray

    """
    num_groups = None
    if groups:
        num_groups = groups[0].shape[1]
        check_input_sample(input_sample, num_groups, N)
    else:
        check_input_sample(input_sample, num_params, N)

    index_list = make_index_list(N, num_params, num_groups)
    distance_matrix = np.zeros((N, N), dtype=np.float32)

    for j in range(N):
        input_1 = input_sample[index_list[j]]
        for k in range(j + 1, N):
            input_2 = input_sample[index_list[k]]

            if local_optimization is True:
                distance_matrix[j, k] = compute_distance(input_1, input_2)

            distance_matrix[k, j] = compute_distance(input_1, input_2)
    return distance_matrix


def find_most_distant(input_sample, N, num_params, k_choices, groups=None):
    """
    Finds the 'k_choices' most distant choices from the
    'N' trajectories contained in 'input_sample'

    Returns
    -------
    numpy.ndarray
    """
    # Now evaluate the (N choose k_choices) possible combinations
    if nchoosek(N, k_choices) >= sys.maxsize:
        raise ValueError("Number of combinations is too large")
    number_of_combinations = int(nchoosek(N, k_choices))

    # First compute the distance matrix for each possible pairing
    # of trajectories and store in a shared-memory array
    distance_matrix = compute_distance_matrix(input_sample,
                                              N,
                                              num_params,
                                              groups)

    # Initialise the output array
    chunk = int(1e6)
    if chunk > number_of_combinations:
        chunk = number_of_combinations

    counter = 0
    # Generate a list of all the possible combinations
    # combos = np.array([x for x in combinations(range(N),k_choices)])
    combo_gen = combinations(list(range(N)), k_choices)
    scores = np.zeros(number_of_combinations, dtype=np.float32)
    # Generate the pairwise indices once
    pairwise = np.array([y for y in combinations(list(range(k_choices)), 2)])

    for combos in grouper(chunk, combo_gen):
        scores[(counter * chunk):((counter + 1) * chunk)] \
            = mappable(combos, pairwise, distance_matrix)
        counter += 1
    return scores


def mappable(combos, pairwise, distance_matrix):
    '''
    Obtains scores from the distance_matrix for each pairwise combination
    held in the combos array
    '''
    combos = np.array(combos)
    # Create a list of all pairwise combination for each combo in combos
    combo_list = combos[:, pairwise[:, ]]
    all_distances = distance_matrix[[combo_list[:, :, 1], combo_list[:, :, 0]]]
    new_scores = np.sqrt(np.einsum('ij,ij->i', all_distances, all_distances))
    return new_scores


def find_maximum(scores, N, k_choices):
    """Finds the `k_choices` maximum scores from `scores`

    Arguments
    ---------
    scores : numpy.ndarray
    N : int
    k_choices : int
    """
    if not isinstance(scores, np.ndarray):
        raise TypeError("Scores input is not a numpy array")

    index_of_maximum = int(scores.argmax())
    maximum_combo = nth(combinations(
        list(range(N)), k_choices), index_of_maximum, None)
    return sorted(maximum_combo)


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
    """Returns the nth item or a default value

    Arguments
    ---------
    iterable : iterable
    n : int
    default : default=None
        The default value to return
    """

    if type(n) != int:
        raise TypeError("n is not an integer")

    return next(islice(iterable, n, None), default)


def make_index_list(N, num_params, groups=None):
    """

    Returns
    -------
    list of numpy.ndarray
    """
    if groups is None:
        groups = num_params

    index_list = []
    for j in range(N):
        index_list.append(np.arange(groups + 1) + j * (groups + 1))
    return index_list


def compile_output(input_sample, N, num_params, maximum_combo, groups=None):
    """
    """

    if groups is None:
        groups = num_params

    check_input_sample(input_sample, groups, N)

    index_list = make_index_list(N, num_params, groups)

    output = np.zeros((np.size(maximum_combo) * (groups + 1), num_params))
    for counter, x in enumerate(maximum_combo):
        output[index_list[counter]] = np.array(input_sample[index_list[x]])
    return output
