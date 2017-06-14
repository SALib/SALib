from __future__ import division

from nose.tools import raises
from numpy.testing import assert_equal, assert_array_equal, \
    assert_almost_equal, assert_allclose

import numpy as np

from SALib.sample.morris_util import generate_P_star, \
    compute_B_star, \
    compute_delta, \
    generate_trajectory, \
    compute_distance, \
    compute_distance_matrix, \
    find_maximum, \
    make_index_list, \
    check_input_sample


def setup():
    input_2 = [[0, 1 / 3.], [2 / 3., 1 / 3.], [2 / 3., 1.]]
    input_1 = [[0, 1 / 3.], [0, 1.], [2 / 3., 1.]]
    input_3 = [[2 / 3., 0], [2 / 3., 2 / 3.], [0, 2 / 3.]]
    input_4 = [[1 / 3., 1.], [1., 1.], [1, 1 / 3.]]
    input_5 = [[1 / 3., 1.], [1 / 3., 1 / 3.], [1, 1 / 3.]]
    input_6 = [[1 / 3., 2 / 3.], [1 / 3., 0], [1., 0]]
    return np.concatenate([input_1, input_2, input_3, input_4, input_5,
                           input_6])


def test_generate_P_star():
    '''
    Matrix P* - size (g * g) - describes order in which groups move
    each row contains one element equal to 1, all others are 0
    no two columns have 1s in the same position
    '''
    for i in range(1, 100):
        output = generate_P_star(i)
        if np.any(np.sum(output, 0) != np.ones(i)):
            raise AssertionError("Not legal P along axis 0")
        elif np.any(np.sum(output, 1) != np.ones(i)):
            raise AssertionError("Not legal P along axis 1")


def test_compute_delta():
    fixture = np.arange(2, 10)
    output = [compute_delta(f) for f in fixture]
    desired = np.array([1.00, 0.75, 0.66, 0.62,
                        0.60, 0.58, 0.57, 0.56])
    assert_almost_equal(output, desired, decimal=2)


def test_generate_trajectory():
    # Two groups of three factors
    G = np.array([[1, 0], [0, 1], [0, 1]])
    # Four levels, grid_jump = 2
    num_levels, grid_jump = 4, 2
    output = generate_trajectory(G, num_levels, grid_jump)
    if np.any((output > 1) | (output < 0)):
        raise AssertionError("Bound not working")
    assert_equal(output.shape[0], 3)
    assert_equal(output.shape[1], 3)


def test_compute_B_star():
    '''
    Tests for expected output

    Taken from example 3.2 in Saltelli et al. (2008) pg 122
    '''

    k = 3
    g = 2

    x_star = np.matrix(np.array([1. / 3, 1. / 3, 0.]))
    J = np.matrix(np.ones((g + 1, k)))
    G = np.matrix('1,0;0,1;0,1')
    D_star = np.matrix('1,0,0;0,-1,0;0,0,1')
    P_star = np.matrix('1,0;0,1')
    delta = 2. / 3
    B = np.matrix(np.tril(np.ones([g + 1, g], dtype=int), -1))

    desired = np.array([[1. / 3, 1, 0], [1, 1, 0], [1, 1. / 3, 2. / 3]])

    output = compute_B_star(J, x_star, delta, B, G, P_star, D_star)
    assert_array_equal(output, desired)


def test_distance():
    '''
    Tests the computation of the distance of two trajectories
    '''
    input_1 = np.matrix([[0, 1 / 3.], [0, 1.], [2 / 3., 1.]], dtype=np.float32)
    input_3 = np.matrix([[2 / 3., 0], [2 / 3., 2 / 3.],
                         [0, 2 / 3.]], dtype=np.float32)
    output = compute_distance(input_1, input_3)
    assert_allclose(output, 6.18, atol=1e-2)


def test_distance_of_identical_matrices_is_min():
    input_1 = np.matrix([[1., 1.],
                         [1., 0.33333333],
                         [0.33333333, 0.33333333]])
    input_2 = input_1.copy()
    actual = compute_distance(input_1, input_2)
    desired = 0
    assert_allclose(actual, desired, atol=1e-2)


def test_distance_fail_with_difference_size_ip():
    input_1 = np.matrix([[0, 1 / 3.], [0, 1.]], dtype=np.float32)
    input_3 = np.matrix([[2 / 3., 0], [2 / 3., 2 / 3.],
                         [0, 2 / 3.]], dtype=np.float32)
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
    assert_allclose(output, expected, rtol=1e-2)


def test_compute_distance_matrix_local():
    '''
    Tests that a distance matrix is computed correctly for the local distance
    optimization.
    The only change is that the local method needs the upper triangle of
    the distance matrix instead of the lower one.

    This is for an input of six trajectories and two parameters
    '''
    sample_inputs = setup()
    output = compute_distance_matrix(
        sample_inputs, 6, 2, local_optimization=True)
    expected = np.zeros((6, 6), dtype=np.float32)
    expected[0, :] = [0,    5.50, 6.18, 6.89, 6.18, 7.52]
    expected[1, :] = [5.50, 0,    5.31, 6.18, 5.31, 5.99]
    expected[2, :] = [6.18, 5.31, 0,    6.57, 5.41, 5.52]
    expected[3, :] = [6.89, 6.18, 6.57, 0,    5.50, 7.31]
    expected[4, :] = [6.18, 5.31, 5.41, 5.5,  0,    5.77]
    expected[5, :] = [7.52, 5.99, 5.52, 7.31, 5.77, 0]
    assert_allclose(output, expected, rtol=1e-2)


def test_find_maximum():
    scores = np.array(range(15))
    k_choices = 4
    N = 6
    output = find_maximum(scores, N, k_choices)
    expected = [2, 3, 4, 5]
    assert_equal(output, expected)


def test_make_index_list():
    N = 4
    num_params = 2
    groups = None
    actual = make_index_list(N, num_params, groups)
    desired = [np.array([0, 1, 2]), np.array([3, 4, 5]),
               np.array([6, 7, 8]), np.array([9, 10, 11])]
    assert_equal(desired, actual)


def test_make_index_list_with_groups():
    N = 4
    num_params = 3
    groups = 2
    actual = make_index_list(N, num_params, groups)
    desired = [np.array([0, 1, 2]), np.array([3, 4, 5]),
               np.array([6, 7, 8]), np.array([9, 10, 11])]
    assert_equal(desired, actual)


@raises(AssertionError)
def test_check_input_sample_N():
    input_sample = setup()
    num_params = 4
    N = 5
    check_input_sample(input_sample, num_params, N)


@raises(AssertionError)
def test_check_input_sample_num_vars():
    input_sample = setup()
    num_params = 3
    N = 6
    check_input_sample(input_sample, num_params, N)


@raises(AssertionError)
def test_check_input_sample_range():
    input_sample = setup()
    input_sample *= 100
    num_params = 4
    N = 6
    check_input_sample(input_sample, num_params, N)
