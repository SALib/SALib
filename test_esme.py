from nose.tools import assert_almost_equal, assert_equal
from esme import compute_distance, compute_distance_matrix, \
    find_most_distant
import numpy as np


def test_distance():
    input_1 = np.matrix([[0, 1/3.], [0, 1.], [2/3., 1.]], dtype=np.float16)
    input_3 = np.matrix([[2/3., 0], [2/3., 2/3.], [0, 2/3.]], dtype=np.float16)
    output = compute_distance(input_1, input_3)
    assert_almost_equal(output, 6.18, places=2)


def test_distance_fail_with_difference_size_ip():
    input_1 = np.matrix([[0, 1/3.], [0, 1.], [2/3., 1.]], dtype=np.float16)
    input_3 = np.matrix([[2/3., 0], [2/3., 2/3.], [0, 2/3.]], dtype=np.float16)
    try:
        compute_distance(input_1, input_3)
    except:
        pass
    else:
        raise AssertionError("Difference size matrices did not trigger error")


def test_compute_distance_matrix():
    sample_inputs = setUp()
    output = compute_distance_matrix(sample_inputs, 6, 2)
    expected = np.zeros((6, 6))
    expected[1, :] = [5.5, 0, 0, 0, 0, 0]
    expected[2, :] = [6.18, 5.31, 0, 0, 0, 0]
    expected[3, :] = [6.89, 6.18, 6.57, 0, 0, 0]
    expected[4, :] = [6.18, 5.31, 5.41, 5.5, 0, 0]
    expected[5, :] = [7.52, 5.99, 5.52, 7.31, 5.77, 0]
    np.testing.assert_array_almost_equal(output, expected, 2)


def setUp():
    input_2 = [[0, 1/3.], [2/3., 1/3.], [2/3., 1.]]
    input_1 = [[0, 1/3.], [0, 1.], [2/3., 1.]]
    input_3 = [[2/3., 0], [2/3., 2/3.], [0, 2/3.]]
    input_4 = [[1/3., 1.], [1., 1.], [1, 1/3.]]
    input_5 = [[1/3., 1.], [1/3., 1/3.], [1, 1/3.]]
    input_6 = [[1/3., 2/3.], [1/3., 0], [1., 0]]
    return np.concatenate([input_1, input_2, input_3, input_4, input_5,
                          input_6])


def test_find_most_distant():
    sample_inputs = setUp()
    N = 6
    num_params = 2
    k_choices = 4
    output = find_most_distant(sample_inputs, N, num_params, k_choices)
    expected = (0, 2, 3, 5)
    assert_equal(output, expected)
