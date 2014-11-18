from nose.tools import assert_almost_equal, assert_equal, raises
from ..test_functions.Sobol_G import evaluate
import numpy as np


def test_Sobol_G():
    desired = np.array([4.0583])

    parameter_values = np.zeros((1,8))

    output = evaluate(parameter_values)

    np.testing.assert_almost_equal(output, desired, decimal=4)


@raises(ValueError)
def test_Sobol_G_raises_error_if_values_wrong_size():
    """
    Tests that a value error is raised if the Sobol G function is called with
    the wrong number of variables
    """
    evaluate(np.array([1, 2, 3, 4, 5, 6, 7]))


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
    evaluate(np.array([0, -1, -.02, 1, 1, -01, -0, -12]))


@raises(TypeError)
def test_Sobol_G_raises_error_if_values_not_numpy_array():
    """
    Tests that a type error is raised if the Sobol G function is called with
    values argument not as a numpy array.
    """
    fixture = [list(range(8)), str(01234567)]
    for x in fixture:
        evaluate(x)
