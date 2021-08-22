from pytest import raises

from numpy.testing import assert_allclose
import numpy as np

from SALib.test_functions.Sobol_G import evaluate, _total_variance, \
    _partial_first_order_variance, \
    sensitivity_index, \
    total_sensitivity_index


def test_Sobol_G():
    '''
    '''
    parameter_values = np.zeros((1, 8))
    actual = evaluate(parameter_values)
    expected = np.array([4.0583])
    assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_Sobol_G_raises_error_if_values_wrong_size():
    """
    Tests that a value error is raised if the Sobol G function is called with
    the wrong number of variables
    """
    a = [1, 2, 3, 4, 5, 6, 7, 8]
    with raises(ValueError):
        evaluate(np.array([1, 2, 3, 4, 5, 6, 7]), a)


def test_Sobol_G_raises_error_if_values_gt_one():
    """
    Tests that a value error is raised if the Sobol G function is called with
    values greater than one
    """
    with raises(ValueError):
        evaluate(np.array([0, 1, .02, 0.23, 1.234, 0.02848848, 0, 0.78]))


def test_Sobol_G_raises_error_if_values_lt_zero():
    """
    Tests that a value error is raised if the Sobol G function is called with
    values less than zero.
    """
    with raises(ValueError):
        evaluate(np.array([0, -1, -.02, 1, 1, -0.1, -0, -12]))


def test_Sobol_G_raises_error_if_values_not_numpy_array():
    """
    Tests that a type error is raised if the Sobol G function is called with
    values argument not as a numpy array.
    """
    fixture = [list(range(8)), str(12345678)]
    for x in fixture:
        with raises(TypeError):
            evaluate(x)


def test_total_variance():

    a = np.array([78, 12, 0.5, 2, 97, 33])
    actual = _total_variance(a)
    expected = 0.19347

    assert_allclose(actual, expected, rtol=1e-4)


def test_partial_first_order_variance():

    a = np.array([78, 12, 0.5, 2, 97, 33])
    actual = _partial_first_order_variance(a)
    expected = (len(a),)

    assert a.shape == expected

    expected = np.array([0.000053, 0.001972, 0.148148,
                         0.037037, 0.000035, 0.000288])

    assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_sensitivity_index():
    a = np.array([78, 12, 0.5, 2, 97, 33])
    actual = sensitivity_index(a)
    expected = np.array([0.000276, 0.010195, 0.765743,
                         0.191436, 0.000179, 0.001490])
    assert_allclose(actual, expected, atol=1e-2, rtol=1e-6)


def test_total_sensitivity_index():
    a = np.array([78, 12, 0.5, 2, 97, 33])

    actual = total_sensitivity_index(a)
    expected = np.array([3.294577e-04, 1.214324e-02, 7.959698e-01, 2.203131e-01,
                         2.140966e-04, 1.778255e-03])


    assert_allclose(actual, expected, atol=1e-2, rtol=1e-6)


def test_modified_Sobol_G():
    parameter_values = np.zeros((1, 8))
    delta_values = np.ones_like(parameter_values)
    alpha_values = np.array([2]*8)
    actual = evaluate(parameter_values, delta=delta_values, alpha=alpha_values)
    expected = np.array([10.6275])
    assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


def test_modified_Sobol_G_error_if_type_wrong():

    parameter_values = np.zeros((1, 8))
    delta_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    expected_err = "The argument `delta` must be given as a numpy ndarray"
    with raises(TypeError, match=expected_err):
        evaluate(parameter_values, delta=delta_values)

    alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    expected_err = "The argument `alpha` must be given as a numpy ndarray"
    with raises(TypeError, match=expected_err):
        evaluate(parameter_values, alpha=alpha_values)


def test_modified_Sobol_G_error_if_value_beyond_range():
    parameter_values = np.zeros((1, 8))
    delta_values = np.array([-0.5, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 1.8])

    expected_err = "Sobol G function called with delta values less than zero or greater than one"
    with raises(ValueError, match=expected_err):
        evaluate(parameter_values, delta=delta_values)

    alpha_values = np.array([0, -0.2, -0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    expected_err = "Sobol G function called with alpha values less than or equal to zero"
    with raises(ValueError, match=expected_err):
        evaluate(parameter_values, alpha=alpha_values)


def test_modified_partial_first_order_variance():

    a = np.array([78, 12, 0.5, 2, 97, 33])
    alpha = np.array([1, 2, 15, 0.6, 8, 48])
    actual = _partial_first_order_variance(a, alpha)
    expected = (len(a),)

    assert a.shape == expected

    expected = np.array([5.34102441e-05, 4.73372781e-03, 3.22580645e+00,
                         1.81818182e-02, 3.91993532e-04, 2.05472122e-02])

    assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)

