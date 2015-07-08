from nose.tools import assert_almost_equal, assert_equal, raises

from numpy.testing import assert_allclose
import numpy as np

from ..test_functions.Sobol_G import evaluate, total_variance, \
                                     partial_first_order_variance, \
                                     sensitivity_index, \
                                     total_sensitivity_index

def test_Sobol_G():
    '''
    '''
    parameter_values = np.zeros((1, 8))
    actual = evaluate(parameter_values)
    expected = np.array([4.0583])
    assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)


@raises(ValueError)
def test_Sobol_G_raises_error_if_values_wrong_size():
    """
    Tests that a value error is raised if the Sobol G function is called with
    the wrong number of variables
    """
    a = [1, 2, 3, 4, 5, 6, 7, 8]
    evaluate(np.array([1, 2, 3, 4, 5, 6, 7]), a)


@raises(ValueError)
def test_Sobol_G_raises_error_if_values_gt_one():
    """
    Tests that a value error is raised if the Sobol G function is called with
    values greater than one
    """
    evaluate(np.array([0, 1, .02, 0.23, 1.234, 0.02848848, 0, 0.78]))


@raises(ValueError)
def test_Sobol_G_raises_error_if_values_lt_zero():
    """
    Tests that a value error is raised if the Sobol G function is called with
    values less than zero.
    """
    evaluate(np.array([0, -1, -.02, 1, 1, -0.1, -0, -12]))


@raises(TypeError)
def test_Sobol_G_raises_error_if_values_not_numpy_array():
    """
    Tests that a type error is raised if the Sobol G function is called with
    values argument not as a numpy array.
    """
    fixture = [list(range(8)), str(12345678)]
    for x in fixture:
        evaluate(x)


def test_total_variance():
    
    a = np.array([78, 12, 0.5, 2, 97, 33])
    actual = total_variance(a)
    expected = 0.19347
    
    assert_allclose(actual, expected, rtol=1e-4)    


def test_partial_first_order_variance():
    
    a = np.array([78, 12, 0.5, 2, 97, 33])
    actual = partial_first_order_variance(a)
    expected = (len(a),)
    
    assert_equal(a.shape, expected)
    
    expected = np.array([0.000053, 0.001972, 0.148148, 0.037037, 0.000035, 0.000288])
    
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
    
    expected = np.array([0.030956547, 0.040875287, 0.796423551,
                         0.222116249, 0.030859879, 0.032170899])
    
    assert_allclose(actual, expected, atol=1e-2, rtol=1e-6)             
