from __future__ import division
import numpy as np
from numpy.testing import assert_equal
from nose.tools import assert_almost_equal, raises
from nose import with_setup
from ..sample.morris_util import generate_P_star,\
                                compute_B_star, \
                                compute_delta, \
                                generate_trajectory, \
                                compute_distance, \
                                compute_distance_matrix, \
                                find_most_distant, find_maximum, \
                                make_index_list


def setup():
    input_2 = [[0, 1/3.], [2/3., 1/3.], [2/3., 1.]]
    input_1 = [[0, 1/3.], [0, 1.], [2/3., 1.]]
    input_3 = [[2/3., 0], [2/3., 2/3.], [0, 2/3.]]
    input_4 = [[1/3., 1.], [1., 1.], [1, 1/3.]]
    input_5 = [[1/3., 1.], [1/3., 1/3.], [1, 1/3.]]
    input_6 = [[1/3., 2/3.], [1/3., 0], [1., 0]]
    return np.concatenate([input_1, input_2, input_3, input_4, input_5,
                          input_6])


def test_generate_P_star():
    '''
    Matrix P* - size (g * g) - describes order in which groups move
    each row contains one element equal to 1, all others are 0
    no two columns have 1s in the same position
    '''
    for i in range(1,100):
        output = generate_P_star(i)
        if np.any(np.sum(output,0) != np.ones(i)):
            raise AssertionError("Not legal P along axis 0")
        elif np.any(np.sum(output, 1) != np.ones(i)):
            raise AssertionError("Not legal P along axis 1")


def test_compute_delta():
    fixture = np.arange(2,10)
    output = [compute_delta(f) for f in fixture]
    desired = np.array([1.00, 0.75, 0.66, 0.62,
                        0.60, 0.58, 0.57, 0.56])
    np.testing.assert_almost_equal(output, desired, decimal=2)


def test_generate_trajectory():
    # Two groups of three factors
    G = np.array([[1,0],[0,1],[0,1]])
    # Four levels, grid_jump = 2
    num_levels, grid_jump = 4, 2
    output = generate_trajectory(G, num_levels, grid_jump)
    if np.any((output > 1) | (output < 0)):
        raise AssertionError("Bound not working")
    assert_equal(output.shape[0],3)
    assert_equal(output.shape[1],3)


def test_compute_B_star():
    '''
    Tests for expected output

    Taken from example 3.2 in Saltelli et al. (2008) pg 122
    '''

    k = 3
    g = 2

    x_star = np.matrix(np.array([1./3, 1./3, 0.]))
    J = np.matrix(np.ones((g + 1, k)))
    G = np.matrix('1,0;0,1;0,1')
    D_star = np.matrix('1,0,0;0,-1,0;0,0,1')
    P_star = np.matrix('1,0;0,1')
    delta = 2./3
    B = np.matrix(np.tril(np.ones([g + 1, g], dtype=int), -1))

    desired = np.array([[1./3, 1, 0],[1, 1, 0],[1, 1./3, 2./3]])

    output = compute_B_star(J, x_star, delta, B, G, P_star, D_star)
    np.testing.assert_array_equal(output, desired)


def test_distance():
    '''
    Tests the computation of the distance of two trajectories
    '''
    input_1 = np.matrix([[0, 1/3.], [0, 1.], [2/3., 1.]], dtype=np.float32)
    input_3 = np.matrix([[2/3., 0], [2/3., 2/3.], [0, 2/3.]], dtype=np.float32)
    output = compute_distance(input_1, input_3)
    assert_almost_equal(output, 6.18, places=2)


def test_distance_fail_with_difference_size_ip():
    input_1 = np.matrix([[0, 1/3.], [0, 1.]], dtype=np.float32)
    input_3 = np.matrix([[2/3., 0], [2/3., 2/3.], [0, 2/3.]], dtype=np.float32)
    try:
        compute_distance(input_1, input_3, 2)
    except:
        pass
    else:
        raise AssertionError("Different size matrices did not trigger error")


def test_compute_distance_matrix():
    '''
    Tests that a distance matrix is computed correctly

    for an input of six trajectories and two parameters
    '''
    sample_inputs = setup()
    output = compute_distance_matrix(sample_inputs, 6, 2)
    expected = np.zeros((6, 6), dtype=np.float32)
    expected[1, :] = [5.50, 0, 0, 0, 0, 0]
    expected[2, :] = [6.18, 5.31, 0, 0, 0, 0]
    expected[3, :] = [6.89, 6.18, 6.57, 0, 0, 0]
    expected[4, :] = [6.18, 5.31, 5.41, 5.5, 0, 0]
    expected[5, :] = [7.52, 5.99, 5.52, 7.31, 5.77, 0]
    np.testing.assert_allclose(output, expected, rtol=1e-2)


def test_combo_from_find_most_distant():
    '''
    Tests whether the correct combination is picked from the fixture drawn
    from Saltelli et al. 2008, in the solution to exercise 3a,
    Chapter 3, page 134.
    '''
    sample_inputs = setup()
    N = 6
    num_params = 2
    k_choices = 4
    scores = find_most_distant(sample_inputs, N, num_params, k_choices)
    output = find_maximum(scores, N, k_choices)
    expected = [0, 2, 3, 5]    # trajectories 1, 3, 4, 6
    np.testing.assert_equal(output, expected)


def test_scores_from_find_most_distant():
    '''
    Checks whether array of scores from (6 4) is correct.

    Data is derived from Saltelli et al. 2008,
    in the solution to exercise 3a, Chapter 3, page 134.

    '''
    sample_inputs = setup()
    N = 6
    num_params = 2
    k_choices = 4
    output = find_most_distant(sample_inputs, N, num_params, k_choices)
    expected = np.array([15.022, 13.871, 14.815, 14.582, 16.178, 14.912, 15.055, 16.410,
                15.685, 16.098, 14.049, 15.146, 14.333, 14.807, 14.825],
                dtype=np.float32)

    np.testing.assert_allclose(output, expected, rtol=1e-1, atol=1e-2)


def test_find_maximum():
    scores = np.array(range(15))
    k_choices = 4
    N = 6
    output = find_maximum(scores, N, k_choices)
    expected = [2, 3, 4, 5]
    assert_equal(output, expected)


def test_catch_combos_too_large():
    N = 1e6
    k_choices = 4
    num_params = 2
    input_sample = np.random.random_sample((N,num_params))

    try:
        find_most_distant(input_sample, N, num_params, k_choices)
    except:
        pass
    else:
        raise AssertionError("Test did not fail when number of \
                             combinations exceeded system size")
        

def test_make_index_list():
    N = 4
    num_params = 2
    groups = None
    actual = make_index_list(N, num_params, groups)
    desired = [np.array([0,1,2]),np.array([3,4,5]),np.array([6,7,8]),np.array([9,10,11])]
    assert_equal(desired, actual)
    

def test_make_index_list_with_groups():
    N = 4
    num_params = 3
    groups = 2
    actual = make_index_list(N, num_params, groups)
    desired = [np.array([0,1,2]),np.array([3,4,5]),np.array([6,7,8]),np.array([9,10,11])]
    assert_equal(desired, actual)