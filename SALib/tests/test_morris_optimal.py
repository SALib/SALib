from nose.tools import assert_almost_equal, assert_equal, raises
from nose import with_setup
from ..sample.morris_optimal import compute_distance, compute_distance_matrix, \
    find_most_distant, find_maximum, find_optimum_trajectories
from ..sample.morris_oat import sample
from ..util import read_param_file
import numpy as np


def setup():
    input_2 = [[0, 1/3.], [2/3., 1/3.], [2/3., 1.]]
    input_1 = [[0, 1/3.], [0, 1.], [2/3., 1.]]
    input_3 = [[2/3., 0], [2/3., 2/3.], [0, 2/3.]]
    input_4 = [[1/3., 1.], [1., 1.], [1, 1/3.]]
    input_5 = [[1/3., 1.], [1/3., 1/3.], [1, 1/3.]]
    input_6 = [[1/3., 2/3.], [1/3., 0], [1., 0]]
    return np.concatenate([input_1, input_2, input_3, input_4, input_5,
                          input_6])


def test_distance():
    '''
    Tests the computation of the distance of two trajectories
    '''
    input_1 = np.matrix([[0, 1/3.], [0, 1.], [2/3., 1.]], dtype=np.float32)
    input_3 = np.matrix([[2/3., 0], [2/3., 2/3.], [0, 2/3.]], dtype=np.float32)
    output = compute_distance(input_1, input_3, 2)
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


def test_find_optimum_trajectories():
    input_1 = [[0, 1/3.], [0, 1.], [2/3., 1.]]
    input_2 = [[0, 1/3.], [2/3., 1/3.], [2/3., 1.]]
    input_3 = [[2/3., 0], [2/3., 2/3.], [0, 2/3.]]
    input_4 = [[1/3., 1.], [1., 1.], [1, 1/3.]]
    input_5 = [[1/3., 1.], [1/3., 1/3.], [1, 1/3.]]
    input_6 = [[1/3., 2/3.], [1/3., 0], [1., 0]]
    input_sample = np.concatenate([input_1, input_2, input_3, input_4, input_5,
                          input_6])
    N = 6
    num_params = 2
    k_choices = 4
    output = find_optimum_trajectories(input_sample, N, num_params, k_choices)
    expected = np.concatenate([input_1, input_3, input_4, input_6])
    np.testing.assert_equal(output, expected)


def setup_function():
    filename = "SALib/tests/test_params.txt"
    with open(filename, "w") as ofile:
         ofile.write("Test1 0.0 100.0\n")
         ofile.write("Test2 5.0 51.0\n")


@with_setup(setup_function)
def test_readfile():

    filename = "SALib/tests/test_params.txt"
    pf = read_param_file(filename)

    assert_equal(pf['bounds'], [[0, 100], [5, 51]])
    assert_equal(pf['num_vars'], 2)
    assert_equal(pf['names'], ['Test1', 'Test2'])


def test_find_maximum():
    scores = np.array(range(15))
    k_choices = 4
    N = 6
    output = find_maximum(scores, N, k_choices)
    expected = (2, 3, 4, 5)
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


@with_setup(setup_function)
@raises(ValueError)
def test_catch_inputs_not_in_zero_one_range():
    filename = "SALib/tests/test_params.txt"
    num_levels = 4
    grid_jump = 2
    num_params = 2
    k_choices = 4
    N = 10
    input_sample = sample(N, filename, num_levels, grid_jump)
    find_optimum_trajectories(input_sample, N, num_params, k_choices)
