from pytest import raises

from numpy.testing import assert_allclose
import numpy as np

from SALib.test_functions.Sobol_G import (evaluate, _total_variance,
                                          _partial_first_order_variance,
                                          sensitivity_index,
                                          total_sensitivity_index)

from SALib import ProblemSpec


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


def test_oakley_results():
    """Test Oakley test function results against analytic values for
    first-order sensitivities provided in [1] (see Table 1).

    Ensure analytic results are within estimated CI.

    References
    ----------
    .. [1] Oakley, J.E., O’Hagan, A., 2004.
           Probabilistic sensitivity analysis of complex models: a Bayesian approach.
           Journal of the Royal Statistical Society: Series B (Statistical Methodology) 66, 751–769.
           https://doi.org/10.1111/j.1467-9868.2004.05304.x
    """
    from SALib.test_functions import oakley2004

    analytic = np.array([0.00156,
                         0.000186,
                         0.001307,
                         0.003045,
                         0.002905,
                         0.023035,
                         0.024151,
                         0.026517,
                         0.046036,
                         0.014945,
                         0.101823,
                         0.135708,
                         0.101989,
                         0.105169,
                         0.122818])

    # Raw values taken from: http://www.jeremy-oakley.staff.shef.ac.uk/psa_example.txt
    M = np.array([
        -2.25E-02, -1.85E-01, 1.34E-01, 3.69E-01, 1.72E-01, 1.37E-01, -4.40E-01, -8.14E-02, 7.13E-01, -4.44E-01, 5.04E-01, -2.41E-02, -4.59E-02, 2.17E-01, 5.59E-02,
        2.57E-01, 5.38E-02, 2.58E-01, 2.38E-01, -5.91E-01, -8.16E-02, -2.87E-01, 4.16E-01, 4.98E-01, 8.39E-02, -1.11E-01, 3.32E-02, -1.40E-01, -3.10E-02, -2.23E-01,
        -5.60E-02, 1.95E-01, 9.55E-02, -2.86E-01, -1.44E-01, 2.24E-01, 1.45E-01, 2.90E-01, 2.31E-01, -3.19E-01, -2.90E-01, -2.10E-01, 4.31E-01, 2.44E-02, 4.49E-02,
        6.64E-01, 4.31E-01, 2.99E-01, -1.62E-01, -3.15E-01, -3.90E-01, 1.77E-01, 5.80E-02, 1.72E-01, 1.35E-01, -3.53E-01, 2.51E-01, -1.88E-02, 3.65E-01, -3.25E-01,
        -1.21E-01, 1.25E-01, 1.07E-01, 4.66E-02, -2.17E-01, 1.95E-01, -6.55E-02, 2.44E-02, -9.68E-02, 1.94E-01, 3.34E-01, 3.13E-01, -8.36E-02, -2.53E-01, 3.73E-01,
        -2.84E-01, -3.28E-01, -1.05E-01, -2.21E-01, -1.37E-01, -1.44E-01, -1.15E-01, 2.24E-01, -3.04E-02, -5.15E-01, 1.73E-02, 3.90E-02, 3.61E-01, 3.09E-01, 5.00E-02,
        -7.79E-02, 3.75E-03, 8.87E-01, -2.66E-01, -7.93E-02, -4.27E-02, -1.87E-01, -3.56E-01, -1.75E-01, 8.87E-02, 4.00E-01, -5.60E-02, 1.37E-01, 2.15E-01, -1.13E-02,
        -9.23E-02, 5.92E-01, 3.13E-02, -3.31E-02, -2.43E-01, -9.98E-02, 3.45E-02, 9.51E-02, -3.38E-01, 6.39E-03, -6.12E-01, 8.13E-02, 8.87E-01, 1.43E-01, 1.48E-01,
        -1.32E-01, 5.29E-01, 1.27E-01, 4.51E-02, 5.84E-01, 3.73E-01, 1.14E-01, -2.95E-01, -5.70E-01, 4.63E-01, -9.41E-02, 1.40E-01, -3.86E-01, -4.49E-01, -1.46E-01,
        5.81E-02, -3.23E-01, 9.31E-02, 7.24E-02, -5.69E-01, 5.26E-01, 2.37E-01, -1.18E-02, 7.18E-02, 7.83E-02, -1.34E-01, 2.27E-01, 1.44E-01, -4.52E-01, -5.56E-01,
        6.61E-01, 3.46E-01, 1.41E-01, 5.19E-01, -2.80E-01, -1.60E-01, -6.84E-02, -2.04E-01, 6.97E-02, 2.31E-01, -4.44E-02, -1.65E-01, 2.16E-01, 4.27E-03, -8.74E-02,
        3.16E-01, -2.76E-02, 1.34E-01, 1.35E-01, 5.40E-02, -1.74E-01, 1.75E-01, 6.03E-02, -1.79E-01, -3.11E-01, -2.54E-01, 2.58E-02, -4.30E-01, -6.23E-01, -3.40E-02,
        -2.90E-01, 3.41E-02, 3.49E-02, -1.21E-01, 2.60E-02, -3.35E-01, -4.14E-01, 5.32E-02, -2.71E-01, -2.63E-02, 4.10E-01, 2.66E-01, 1.56E-01, -1.87E-01, 1.99E-02,
        -2.44E-01, -4.41E-01, 1.26E-02, 2.49E-01, 7.11E-02, 2.46E-01, 1.75E-01, 8.53E-03, 2.51E-01, -1.47E-01, -8.46E-02, 3.69E-01, -3.00E-01, 1.10E-01, -7.57E-01,
        4.15E-02, -2.60E-01, 4.64E-01, -3.61E-01, -9.50E-01, -1.65E-01, 3.09E-03, 5.28E-02, 2.25E-01, 3.84E-01, 4.56E-01, -1.86E-01, 8.23E-03, 1.67E-01, 1.60E-01
    ]).reshape(15,15)

    A = np.array([[0.0118, 0.0456, 0.2297, 0.0393, 0.1177, 0.3865, 0.3897, 0.6061, 0.6159, 0.4005, 1.0741, 1.1474, 0.7880, 1.1242, 1.1982],
                  [0.4341, 0.0887, 0.0512, 0.3233, 0.1489, 1.0360, 0.9892, 0.9672, 0.8977, 0.8083, 1.8426, 2.4712, 2.3946, 2.0045, 2.2621],
                  [0.1044, 0.2057, 0.0774, 0.2730, 0.1253, 0.7526, 0.8570, 1.0331, 0.8388, 0.7970, 2.2145, 2.0382, 2.4004, 2.0541, 1.9845]])

    sp = ProblemSpec({
        'names': ['x{}'.format(i) for i in range(1,16)],
        'bounds': [[0.0, 0.835],]*15,
        'dists': ['norm']*15
    })

    (sp.sample_latin(2048)
        .evaluate(evaluate, A, M)
        .analyze_rbd_fast(seed=101, num_resamples=100)
    )

    S1 = sp.analysis.to_df()

    S1["lower"] = S1["S1"] - S1["S1_conf"]
    S1["analytic"] = analytic
    S1["upper"] = S1["S1"] + S1["S1_conf"]

    assert np.all((analytic >= S1["lower"]) & (analytic <= S1["upper"]))
