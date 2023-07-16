# -*- coding: utf-8 -*-

from pytest import raises, fixture

import numpy as np
from numpy.testing import assert_allclose, assert_equal

from SALib.analyze.morris import (
    analyze,
    _compute_mu_star_confidence,
    _compute_elementary_effects,
    _reorganize_output_matrix,
    _compute_grouped_metric,
    _compute_grouped_sigma,
    _check_if_array_of_floats,
)


def test_compute_mu_star_confidence():
    """
    Tests that compute mu_star_confidence is computed correctly
    """

    ee = np.array([[2.52, 2.01, 2.30, 0.66, 0.93, 1.3]], dtype=float)
    num_resamples = 1000
    conf_level = 0.95
    num_vars = 1

    actual = _compute_mu_star_confidence(ee, num_vars, num_resamples, conf_level)
    expected = 0.5
    assert_allclose(actual, expected, atol=1e-01)


def test_analysis_of_morris_results():
    """
    Tests a one-dimensional vector of results

    Taken from the solution to Exercise 4 (p.138) in Saltelli (2008).
    """
    model_input = np.array(
        [
            [0, 1.0 / 3],
            [0, 1],
            [2.0 / 3, 1],
            [0, 1.0 / 3],
            [2.0 / 3, 1.0 / 3],
            [2.0 / 3, 1],
            [2.0 / 3, 0],
            [2.0 / 3, 2.0 / 3],
            [0, 2.0 / 3],
            [1.0 / 3, 1],
            [1, 1],
            [1, 1.0 / 3],
            [1.0 / 3, 1],
            [1.0 / 3, 1.0 / 3],
            [1, 1.0 / 3],
            [1.0 / 3, 2.0 / 3],
            [1.0 / 3, 0],
            [1, 0],
        ],
        dtype=float,
    )

    model_output = np.array(
        [
            0.97,
            0.71,
            2.39,
            0.97,
            2.30,
            2.39,
            1.87,
            2.40,
            0.87,
            2.15,
            1.71,
            1.54,
            2.15,
            2.17,
            1.54,
            2.20,
            1.87,
            1.0,
        ],
        dtype=float,
    )

    problem = {
        "num_vars": 2,
        "names": ["Test 1", "Test 2"],
        "groups": None,
        "bounds": [[0.0, 1.0], [0.0, 1.0]],
    }

    Si = analyze(
        problem,
        model_input,
        model_output,
        num_resamples=1000,
        conf_level=0.95,
        print_to_console=False,
    )

    desired_mu = np.array([0.66, 0.21])
    assert_allclose(
        Si["mu"], desired_mu, rtol=1e-1, err_msg="The values for mu are incorrect"
    )
    desired_mu_star = np.array([1.62, 0.35])
    assert_allclose(
        Si["mu_star"],
        desired_mu_star,
        rtol=1e-2,
        err_msg="The values for mu star are incorrect",
    )
    desired_sigma = np.array([1.79, 0.41])
    assert_allclose(
        Si["sigma"],
        desired_sigma,
        rtol=1e-2,
        err_msg="The values for sigma are incorrect",
    )
    desired_names = ["Test 1", "Test 2"]
    assert_equal(
        Si["names"], desired_names, err_msg="The values for names are incorrect"
    )


def test_analysis_of_morris_results_scaled():
    """
    Tests a one-dimensional vector of results

    Taken from the solution to Exercise 4 (p.138) in Saltelli (2008).
    """
    model_input = np.array(
        [
            [0, 1.0 / 3],
            [0, 1],
            [2.0 / 3, 1],
            [0, 1.0 / 3],
            [2.0 / 3, 1.0 / 3],
            [2.0 / 3, 1],
            [2.0 / 3, 0],
            [2.0 / 3, 2.0 / 3],
            [0, 2.0 / 3],
            [1.0 / 3, 1],
            [1, 1],
            [1, 1.0 / 3],
            [1.0 / 3, 1],
            [1.0 / 3, 1.0 / 3],
            [1, 1.0 / 3],
            [1.0 / 3, 2.0 / 3],
            [1.0 / 3, 0],
            [1, 0],
        ],
        dtype=float,
    )

    model_output = np.array(
        [
            0.97,
            0.71,
            2.39,
            0.97,
            2.30,
            2.39,
            1.87,
            2.40,
            0.87,
            2.15,
            1.71,
            1.54,
            2.15,
            2.17,
            1.54,
            2.20,
            1.87,
            1.0,
        ],
        dtype=float,
    )

    problem = {
        "num_vars": 2,
        "names": ["Test 1", "Test 2"],
        "groups": None,
        "bounds": [[0.0, 1.0], [0.0, 1.0]],
    }

    Si = analyze(
        problem,
        model_input,
        model_output,
        num_resamples=1000,
        conf_level=0.95,
        scaled=True,
        print_to_console=False,
    )

    desired_mu = np.array([0.090389, 0.146679])
    assert_allclose(
        Si["mu"], desired_mu, rtol=1e-3, err_msg="The values for mu are incorrect"
    )
    desired_mu_star = np.array([0.968042, 0.212761])
    assert_allclose(
        Si["mu_star"],
        desired_mu_star,
        rtol=1e-3,
        err_msg="The values for mu star are incorrect",
    )
    desired_sigma = np.array([1.064536844, 0.223856942])
    assert_allclose(
        Si["sigma"],
        desired_sigma,
        rtol=1e-3,
        err_msg="The values for sigma are incorrect",
    )
    desired_names = ["Test 1", "Test 2"]
    assert_equal(
        Si["names"], desired_names, err_msg="The values for names are incorrect"
    )


def test_conf_level_within_zero_one_bounds():
    ee = [0, 0, 0]
    num_resamples = 2
    conf_level_too_low = -1
    num_vars = 1
    with raises(ValueError):
        _compute_mu_star_confidence(ee, num_vars, num_resamples, conf_level_too_low)
    conf_level_too_high = 2
    with raises(ValueError):
        _compute_mu_star_confidence(ee, num_vars, num_resamples, conf_level_too_high)


@fixture()
def morris_data() -> tuple:
    """Model inputs and outputs for calculating elementary effects"""
    model_inputs = np.array(
        [
            [
                1.64,
                -1.64,
                -1.64,
                0.39,
                -0.39,
                0.39,
                -1.64,
                -1.64,
                -0.39,
                -0.39,
                1.64,
                1.64,
                -0.39,
                0.39,
                1.64,
            ],
            [
                1.64,
                -1.64,
                -1.64,
                0.39,
                -0.39,
                -1.64,
                -1.64,
                -1.64,
                -0.39,
                -0.39,
                1.64,
                1.64,
                -0.39,
                0.39,
                1.64,
            ],
            [
                1.64,
                -1.64,
                -1.64,
                0.39,
                -0.39,
                -1.64,
                -1.64,
                -1.64,
                1.64,
                -0.39,
                1.64,
                1.64,
                -0.39,
                0.39,
                1.64,
            ],
            [
                1.64,
                -1.64,
                -1.64,
                0.39,
                -0.39,
                -1.64,
                -1.64,
                0.39,
                1.64,
                -0.39,
                1.64,
                1.64,
                -0.39,
                0.39,
                1.64,
            ],
            [
                1.64,
                -1.64,
                -1.64,
                -1.64,
                -0.39,
                -1.64,
                -1.64,
                0.39,
                1.64,
                -0.39,
                1.64,
                1.64,
                -0.39,
                0.39,
                1.64,
            ],
            [
                1.64,
                -1.64,
                -1.64,
                -1.64,
                -0.39,
                -1.64,
                -1.64,
                0.39,
                1.64,
                -0.39,
                1.64,
                -0.39,
                -0.39,
                0.39,
                1.64,
            ],
            [
                1.64,
                -1.64,
                -1.64,
                -1.64,
                -0.39,
                -1.64,
                -1.64,
                0.39,
                1.64,
                -0.39,
                1.64,
                -0.39,
                -0.39,
                -1.64,
                1.64,
            ],
            [
                1.64,
                0.39,
                -1.64,
                -1.64,
                -0.39,
                -1.64,
                -1.64,
                0.39,
                1.64,
                -0.39,
                1.64,
                -0.39,
                -0.39,
                -1.64,
                1.64,
            ],
            [
                1.64,
                0.39,
                -1.64,
                -1.64,
                -0.39,
                -1.64,
                -1.64,
                0.39,
                1.64,
                -0.39,
                1.64,
                -0.39,
                -0.39,
                -1.64,
                -0.39,
            ],
            [
                1.64,
                0.39,
                -1.64,
                -1.64,
                -0.39,
                -1.64,
                -1.64,
                0.39,
                1.64,
                1.64,
                1.64,
                -0.39,
                -0.39,
                -1.64,
                -0.39,
            ],
            [
                1.64,
                0.39,
                -1.64,
                -1.64,
                1.64,
                -1.64,
                -1.64,
                0.39,
                1.64,
                1.64,
                1.64,
                -0.39,
                -0.39,
                -1.64,
                -0.39,
            ],
            [
                1.64,
                0.39,
                -1.64,
                -1.64,
                1.64,
                -1.64,
                -1.64,
                0.39,
                1.64,
                1.64,
                -0.39,
                -0.39,
                -0.39,
                -1.64,
                -0.39,
            ],
            [
                1.64,
                0.39,
                -1.64,
                -1.64,
                1.64,
                -1.64,
                0.39,
                0.39,
                1.64,
                1.64,
                -0.39,
                -0.39,
                -0.39,
                -1.64,
                -0.39,
            ],
            [
                1.64,
                0.39,
                0.39,
                -1.64,
                1.64,
                -1.64,
                0.39,
                0.39,
                1.64,
                1.64,
                -0.39,
                -0.39,
                -0.39,
                -1.64,
                -0.39,
            ],
            [
                -0.39,
                0.39,
                0.39,
                -1.64,
                1.64,
                -1.64,
                0.39,
                0.39,
                1.64,
                1.64,
                -0.39,
                -0.39,
                -0.39,
                -1.64,
                -0.39,
            ],
            [
                -0.39,
                0.39,
                0.39,
                -1.64,
                1.64,
                -1.64,
                0.39,
                0.39,
                1.64,
                1.64,
                -0.39,
                -0.39,
                1.64,
                -1.64,
                -0.39,
            ],
        ],
        dtype=float,
    )
    model_outputs = np.array(
        [
            24.9,
            22.72,
            21.04,
            16.01,
            10.4,
            10.04,
            8.6,
            13.39,
            4.69,
            8.02,
            9.98,
            3.75,
            1.33,
            2.59,
            6.37,
            9.99,
        ],
        dtype=float,
    )

    return model_inputs, model_outputs


def test_compute_elementary_effects(morris_data):
    """
    Inputs for elementary effects taken from Exercise 5 from Saltelli (2008).
    See page 140-145.
    `model_inputs` are from trajectory t_1 from table 3.10 on page 141.
    `desired` is equivalent to column t_1 in table 3.12 on page 145.
    """
    model_inputs, model_outputs = morris_data

    delta = 2.0 / 3

    actual = _compute_elementary_effects(model_inputs, model_outputs, 16, delta)
    desired = np.array(
        [
            [-5.67],
            [7.18],
            [1.89],
            [8.42],
            [2.93],
            [3.28],
            [-3.62],
            [-7.55],
            [-2.51],
            [5.00],
            [9.34],
            [0.54],
            [5.43],
            [2.15],
            [13.05],
        ],
        dtype=float,
    )
    assert_allclose(actual, desired, atol=1e-1)


def test_compute_elementary_effects_scaled(morris_data):
    """
    Inputs for elementary effects taken from Exercise 5 from Saltelli (2008).
    See page 140-145.
    `model_inputs` are from trajectory t_1 from table 3.10 on page 141.
    `desired` was manual calculated in an Excel spreadsheet after confirming
    the results from the original paper.
    """
    model_inputs, model_outputs = morris_data

    delta = 2.0 / 3

    actual = _compute_elementary_effects(
        model_inputs, model_outputs, 16, delta, scaling=True
    )
    desired = np.array(
        [
            [-0.181634813],
            [0.345250299],
            [0.071454753],
            [0.352948837],
            [0.137866888],
            [0.076670871],
            [-0.152252439],
            [-0.285251912],
            [-0.080726583],
            [0.240017431],
            [0.419563468],
            [0.024244438],
            [0.12731585],
            [0.101289958],
            [0.63202974],
        ],
        dtype=float,
    )
    assert_allclose(actual, desired, atol=1e-2)


def test_compute_grouped_elementary_effects():

    model_inputs = np.array(
        [
            [
                0.39,
                -0.39,
                -1.64,
                0.39,
                -0.39,
                -0.39,
                0.39,
                0.39,
                -1.64,
                -0.39,
                0.39,
                -1.64,
                1.64,
                1.64,
                1.64,
            ],
            [
                -1.64,
                1.64,
                0.39,
                -1.64,
                1.64,
                1.64,
                -1.64,
                -1.64,
                -1.64,
                1.64,
                0.39,
                -1.64,
                1.64,
                1.64,
                1.64,
            ],
            [
                -1.64,
                1.64,
                0.39,
                -1.64,
                1.64,
                1.64,
                -1.64,
                -1.64,
                0.39,
                1.64,
                -1.64,
                0.39,
                -0.39,
                -0.39,
                -0.39,
            ],
        ]
    )

    model_results = np.array([13.85, -10.11, 1.12])

    problem = {
        "names": [
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
        ],
        "bounds": [[]],
        "groups": (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0],
                ]
            ),
            ["gp1", "gp2"],
        ),
        "num_vars": 15,
    }
    ee = _compute_elementary_effects(model_inputs, model_results, 3, 2.0 / 3)
    mu_star = np.average(np.abs(ee), axis=1)
    actual = _compute_grouped_metric(mu_star, problem["groups"][0].T)
    desired = np.array([16.86, 35.95])
    assert_allclose(actual, desired, atol=1e-1)


def test_compute_elementary_effects_small():
    """
    Computes elementary effects for two variables, over six trajectories with
    four levels.
    """
    model_inputs = np.array(
        [
            [0, 1.0 / 3],
            [0, 1],
            [2.0 / 3, 1],
            [0, 1.0 / 3],
            [2.0 / 3, 1.0 / 3],
            [2.0 / 3, 1],
            [2.0 / 3, 0],
            [2.0 / 3, 2.0 / 3],
            [0, 2.0 / 3],
            [1.0 / 3, 1],
            [1, 1],
            [1, 1.0 / 3],
            [1.0 / 3, 1],
            [1.0 / 3, 1.0 / 3],
            [1, 1.0 / 3],
            [1.0 / 3, 2.0 / 3],
            [1.0 / 3, 0],
            [1, 0],
        ],
        dtype=float,
    )

    model_outputs = np.array(
        [
            0.97,
            0.71,
            2.39,
            0.97,
            2.3,
            2.39,
            1.87,
            2.40,
            0.87,
            2.15,
            1.71,
            1.54,
            2.15,
            2.17,
            1.54,
            2.2,
            1.87,
            1.0,
        ],
        dtype=float,
    )

    delta = 2.0 / 3
    actual = _compute_elementary_effects(model_inputs, model_outputs, 3, delta)
    desired = np.array(
        [
            [2.52, 2.01, 2.30, -0.66, -0.93, -1.30],
            [-0.39, 0.13, 0.80, 0.25, -0.02, 0.51],
        ]
    )
    assert_allclose(actual, desired, atol=1e-0)


def test_reorganize_output_matrix_increased():
    up = np.array(
        [
            [[False, True], [True, False]],
            [[True, False], [False, True]],
            [[False, True], [False, False]],
            [[True, False], [False, False]],
            [[False, False], [True, False]],
            [[False, False], [True, False]],
        ],
        dtype=bool,
    )

    lo = np.array(
        [
            [[False, False], [False, False]],
            [[False, False], [False, False]],
            [[False, False], [True, False]],
            [[False, False], [False, True]],
            [[False, True], [False, False]],
            [[False, True], [False, False]],
        ],
        dtype=bool,
    )

    model_outputs = np.array(
        [
            0.97,
            0.71,
            2.39,
            0.97,
            2.3,
            2.39,
            1.87,
            2.40,
            0.87,
            2.15,
            1.71,
            1.54,
            2.15,
            2.17,
            1.54,
            2.2,
            1.87,
            1.0,
        ],
        dtype=float,
    )
    op_vec = model_outputs.reshape(6, 3)
    actual = _reorganize_output_matrix(op_vec, up, lo)
    desired = np.array(
        [[2.39, 2.3, 2.4, 1.71, 1.54, 1.0], [0.71, 2.39, 2.40, 1.71, 2.15, 2.20]],
        dtype=float,
    )
    assert_allclose(actual, desired, atol=1e-1)


def test_reorganize_output_matrix_decreased():
    up = np.array(
        [
            [[False, True], [True, False]],
            [[True, False], [False, True]],
            [[False, True], [False, False]],
            [[True, False], [False, False]],
            [[False, False], [True, False]],
            [[False, False], [True, False]],
        ],
        dtype=bool,
    )

    lo = np.array(
        [
            [[False, False], [False, False]],
            [[False, False], [False, False]],
            [[False, False], [True, False]],
            [[False, False], [False, True]],
            [[False, True], [False, False]],
            [[False, True], [False, False]],
        ],
        dtype=bool,
    )

    model_outputs = np.array(
        [
            0.97,
            0.71,
            2.39,
            0.97,
            2.3,
            2.39,
            1.87,
            2.40,
            0.87,
            2.15,
            1.71,
            1.54,
            2.15,
            2.17,
            1.54,
            2.2,
            1.87,
            1.0,
        ],
        dtype=float,
    )
    op_vec = model_outputs.reshape(6, 3)
    actual = _reorganize_output_matrix(op_vec, up, lo, increase=False)
    desired = np.array(
        [[0.71, 0.97, 0.87, 2.15, 2.17, 1.87], [0.97, 2.30, 1.87, 1.54, 2.17, 1.87]],
        dtype=float,
    )
    assert_allclose(actual, desired, atol=1e-1)


def test_compute_grouped_metric():
    """
    Computes mu_star for 3 variables grouped into 2 groups
    There are six trajectories.
    """
    group_matrix = np.array([[1, 0], [0, 1], [0, 1]], dtype=int)

    ee = np.array(
        [
            [2.52, 2.01, 2.30, -0.66, -0.93, -1.30],
            [-2.00, 0.13, -0.80, 0.25, -0.02, 0.51],
            [2.00, -0.13, 0.80, -0.25, 0.02, -0.51],
        ]
    )
    mu_star = np.average(np.abs(ee), 1)
    actual = _compute_grouped_metric(mu_star, group_matrix)
    desired = np.array([1.62, 0.62], dtype=float)

    assert_allclose(actual, desired, rtol=1e-1)


def test_compute_grouped_sigma():
    """
    Tests that a value for sigma is returned when the group contains 1 param

    Morris groups do not allow a value for sigma to be computed because it
    requires the use of mu (as opposed to mu_star).

    However, if the group consists of just one parameter, then the sampling
    will be identical to the situation in which no groups are used,
    but only for that parameter.

    An NA should be returned for all other groups (as opposed to 0, which could
    confuse plotting.morris)
    """
    group_matrix = np.array([[1, 0], [0, 1], [0, 1]], dtype=int)

    ee = np.array(
        [
            [2.52, 2.01, 2.30, -0.66, -0.93, -1.30],
            [-2.00, 0.13, -0.80, 0.25, -0.02, 0.51],
            [2.00, -0.13, 0.80, -0.25, 0.02, -0.51],
        ]
    )
    sigma = np.std(ee, axis=1, ddof=1)

    actual = _compute_grouped_sigma(sigma, group_matrix)
    desired = np.array([1.79352911, np.NAN], dtype=float)
    assert_allclose(actual, desired, rtol=1e-1)


def test_check_if_array_of_floats():
    outputs = np.array(
        [
            0.97,
            0.71,
            2.39,
            0.97,
            2.30,
            2.39,
            1.87,
            2.40,
            0.87,
            2.15,
            1.71,
            1.54,
            2.15,
            2.17,
            1.54,
            2.20,
            1.87,
            1.0,
        ],
        dtype=int,
    )

    with raises(ValueError):
        _check_if_array_of_floats(outputs)


def test_doesnot_raise_error_if_floats():

    inputs = np.array(
        [
            [0, 1.0 / 3],
            [0, 1],
            [2.0 / 3, 1],
            [0, 1.0 / 3],
            [2.0 / 3, 1.0 / 3],
            [2.0 / 3, 1],
            [2.0 / 3, 0],
            [2.0 / 3, 2.0 / 3],
            [0, 2.0 / 3],
            [1.0 / 3, 1],
            [1, 1],
            [1, 1.0 / 3],
            [1.0 / 3, 1],
            [1.0 / 3, 1.0 / 3],
            [1, 1.0 / 3],
            [1.0 / 3, 2.0 / 3],
            [1.0 / 3, 0],
            [1, 0],
        ],
        dtype=np.float32,
    )

    outputs = np.array(
        [
            0.97,
            0.71,
            2.39,
            0.97,
            2.30,
            2.39,
            1.87,
            2.40,
            0.87,
            2.15,
            1.71,
            1.54,
            2.15,
            2.17,
            1.54,
            2.20,
            1.87,
            1.0,
        ],
        dtype=np.float32,
    )

    problem = {
        "num_vars": 2,
        "names": ["Test 1", "Test 2"],
        "groups": None,
        "bounds": [[0.0, 1.0], [0.0, 1.0]],
    }

    analyze(problem, inputs, outputs)
