# -*- coding: utf-8 -*-
from __future__ import division
from numpy.testing import assert_allclose, assert_equal
from nose.tools import raises, with_setup
import numpy as np

from ..analyze.morris import analyze, \
    compute_mu_star_confidence, \
    compute_effects_vector,\
    get_increased_values,\
    get_decreased_values, \
    compute_grouped_mu_star


# Fixtures
def setup_results():
    filename = "SALib/tests/test_results.txt"
    results = np.array([0.97, 0.71, 2.39, 0.97, 2.3, 2.39, 1.87, 2.40, 0.87, 2.15, 1.71, 1.54, 2.15, 2.17, 1.54, 2.2, 1.87, 1.0],
                       dtype=np.float)
    np.savetxt(filename, results)


def setup_parameter_file():
    filename = "SALib/tests/test_param_file.txt"
    with open(filename, "w") as ofile:
        ofile.write("Test 1,0,1.0\n")
        ofile.write("Test 2,0,1.0\n")


def setup_input_file():
    filename = "SALib/tests/model_inputs.txt"
    model_inputs = np.array([[0, 1. / 3], [0, 1],       [2. / 3, 1],
                             [0, 1. / 3],   [2. / 3, 1. / 3], [2. / 3, 1],
                             [2. / 3, 0],   [2. / 3, 2. / 3], [0, 2. / 3],
                             [1. / 3, 1],   [1, 1],       [1, 1. / 3],
                             [1. / 3, 1],   [1. / 3, 1. / 3], [1, 1. / 3],
                             [1. / 3, 2. / 3], [1. / 3, 0],   [1, 0]],
                            dtype=np.float)
    np.savetxt(filename, model_inputs)


def setup():
    setup_parameter_file()
    setup_input_file()
    setup_results()


def teardown():
    pass


@with_setup(setup, teardown)
def test_analysis_of_morris_results():
    output_file = "SALib/tests/test_results.txt"
    pfile = "SALib/tests/test_param_file.txt"
    input_file = "SALib/tests/model_inputs.txt"

    Si = analyze(pfile,
                 input_file,
                 output_file,
                 column=0,
                 delim=' ',
                 num_resamples=1000,
                 conf_level=0.95,
                 print_to_console=False)
    ee = np.array([[2.52, 2.01, 2.30, -0.66, -0.93, -1.30],
                   [-0.39, 0.13, 0.80,  0.25, -0.02,  0.51]])
    desired_mu = np.average(ee, 1)
    assert_allclose(Si['mu'], desired_mu, rtol=1e-1)
    desired_mu_star = np.average(np.abs(ee), 1)
    assert_allclose(Si['mu_star'], desired_mu_star, rtol=1e-2)
    desired_sigma = np.std(ee, 1)
    assert_allclose(Si['sigma'], desired_sigma, rtol=1e-2)
    desired_names = ['Test 1', 'Test 2']
    assert_equal(Si['names'], desired_names)


def test_number_results_incorrect():
    pass


#@raises(ValueError)
# def test_if_column_does_not_exist():
#    pass


@raises(ValueError)
def test_conf_level_within_zero_one_bounds():
    ee = [0, 0, 0]
    N = 1
    num_resamples = 2
    conf_level_too_low = -1
    compute_mu_star_confidence(ee, N, num_resamples, conf_level_too_low)
    conf_level_too_high = 2
    compute_mu_star_confidence(ee, N, num_resamples, conf_level_too_high)


def test_compute_elementary_effects_vector():
    model_inputs = np.array([
                            [1.64, -1.64, -1.64, 0.39, -0.39, 0.39, -1.64, -
                                1.64, -0.39, -0.39, 1.64, 1.64, -0.39, 0.39, 1.64],
                            [1.64, -1.64, -1.64, 0.39, -0.39, -1.64, -1.64, -
                                1.64, -0.39, -0.39, 1.64, 1.64, -0.39, 0.39, 1.64],
                            [1.64, -1.64, -1.64, 0.39, -0.39, -1.64, -1.64, -
                                1.64, 1.64, -0.39, 1.64, 1.64, -0.39, 0.39, 1.64],
                            [1.64, -1.64, -1.64, 0.39, -0.39, -1.64, -1.64,
                                0.39, 1.64, -0.39, 1.64, 1.64, -0.39, 0.39, 1.64],
                            [1.64, -1.64, -1.64, -1.64, -0.39, -1.64, -1.64,
                                0.39, 1.64, -0.39, 1.64, 1.64, -0.39, 0.39, 1.64],
                            [1.64, -1.64, -1.64, -1.64, -0.39, -1.64, -1.64,
                                0.39, 1.64, -0.39, 1.64, -0.39, -0.39, 0.39, 1.64],
                            [1.64, -1.64, -1.64, -1.64, -0.39, -1.64, -1.64,
                                0.39, 1.64, -0.39, 1.64, -0.39, -0.39, -1.64, 1.64],
                            [1.64, 0.39, -1.64, -1.64, -0.39, -1.64, -1.64,
                                0.39, 1.64, -0.39, 1.64, -0.39, -0.39, -1.64, 1.64],
                            [1.64, 0.39, -1.64, -1.64, -0.39, -1.64, -1.64, 0.39,
                                1.64, -0.39, 1.64, -0.39, -0.39, -1.64, -0.39],
                            [1.64, 0.39, -1.64, -1.64, -0.39, -1.64, -1.64,
                                0.39, 1.64, 1.64, 1.64, -0.39, -0.39, -1.64, -0.39],
                            [1.64, 0.39, -1.64, -1.64, 1.64, -1.64, -1.64, 0.39,
                                1.64, 1.64, 1.64, -0.39, -0.39, -1.64, -0.39],
                            [1.64, 0.39, -1.64, -1.64, 1.64, -1.64, -1.64, 0.39,
                                1.64, 1.64, -0.39, -0.39, -0.39, -1.64, -0.39],
                            [1.64, 0.39, -1.64, -1.64, 1.64, -1.64, 0.39, 0.39,
                                1.64, 1.64, -0.39, -0.39, -0.39, -1.64, -0.39],
                            [1.64, 0.39, 0.39, -1.64, 1.64, -1.64, 0.39, 0.39,
                                1.64, 1.64, -0.39, -0.39, -0.39, -1.64, -0.39],
                            [-0.39, 0.39, 0.39, -1.64, 1.64, -1.64, 0.39, 0.39,
                                1.64, 1.64, -0.39, -0.39, -0.39, -1.64, -0.39],
                            [-0.39, 0.39, 0.39, -1.64, 1.64, -1.64, 0.39, 0.39, 1.64, 1.64, -0.39, -0.39, 1.64, -1.64, -0.39]],
                            dtype=np.float)
    model_outputs = np.array([24.9, 22.72, 21.04, 16.01, 10.4, 10.04, 8.6, 13.39, 4.69, 8.02, 9.98, 3.75, 1.33, 2.59, 6.37, 9.99],
                             dtype=np.float)
    delta = 2. / 3

    actual = compute_effects_vector(model_inputs, model_outputs, 16, delta)
    desired = np.array([[-5.67], [7.18], [1.89], [8.42], [2.93], [3.28], [-3.62], [-7.55],
                        [-2.51], [5.00], [9.34], [0.54], [5.43], [2.15], [13.05]],
                       dtype=np.float)
    assert_allclose(actual, desired, atol=1e-1)


def test_compute_elementary_effects_vector_small():
    '''
    Computes elementary effects for two variables,
    over six trajectories with four levels.
    '''
    model_inputs = np.array([[0, 1. / 3], [0, 1],       [2. / 3, 1],
                             [0, 1. / 3],   [2. / 3, 1. / 3], [2. / 3, 1],
                             [2. / 3, 0],   [2. / 3, 2. / 3], [0, 2. / 3],
                             [1. / 3, 1],   [1, 1],       [1, 1. / 3],
                             [1. / 3, 1],   [1. / 3, 1. / 3], [1, 1. / 3],
                             [1. / 3, 2. / 3], [1. / 3, 0],   [1, 0]],
                            dtype=np.float)

    model_outputs = np.array([0.97, 0.71, 2.39, 0.97, 2.3, 2.39, 1.87, 2.40, 0.87, 2.15, 1.71, 1.54, 2.15, 2.17, 1.54, 2.2, 1.87, 1.0],
                             dtype=np.float)

    delta = 2. / 3
    actual = compute_effects_vector(model_inputs, model_outputs, 3, delta)
    desired = np.array(
        [[2.52, 2.01, 2.30, -0.66, -0.93, -1.30], [-0.39, 0.13, 0.80, 0.25, -0.02, 0.51]])
    assert_allclose(actual, desired, atol=1e-0)


def test_compute_increased_value_for_ee():
    up = np.array([[[False,  True], [True, False]],
                   [[True, False], [False,  True]],
                   [[False,  True], [False, False]],
                   [[True, False], [False, False]],
                   [[False, False], [True, False]],
                   [[False, False], [True, False]]],
                  dtype=bool)

    lo = np.array([[[False, False], [False, False]],
                   [[False, False], [False, False]],
                   [[False, False], [True, False]],
                   [[False, False], [False,  True]],
                   [[False,  True], [False, False]],
                   [[False,  True], [False, False]]],
                  dtype=bool)

    model_outputs = np.array([0.97, 0.71, 2.39, 0.97, 2.3, 2.39, 1.87, 2.40, 0.87, 2.15, 1.71, 1.54, 2.15, 2.17, 1.54, 2.2, 1.87, 1.0],
                             dtype=np.float)
    op_vec = model_outputs.reshape(6, 3)
    actual = get_increased_values(op_vec, up, lo)
    desired = np.array([[2.39, 2.3, 2.4, 1.71, 1.54, 1.0],
                        [0.71, 2.39, 2.40, 1.71, 2.15, 2.20]],
                       dtype=np.float)
    assert_allclose(actual, desired, atol=1e-1)


def test_compute_decreased_value_for_ee():
    up = np.array([[[False,  True], [True, False]],
                   [[True, False], [False,  True]],
                   [[False,  True], [False, False]],
                   [[True, False], [False, False]],
                   [[False, False], [True, False]],
                   [[False, False], [True, False]]],
                  dtype=bool)

    lo = np.array([[[False, False], [False, False]],
                   [[False, False], [False, False]],
                   [[False, False], [True, False]],
                   [[False, False], [False,  True]],
                   [[False,  True], [False, False]],
                   [[False,  True], [False, False]]],
                  dtype=bool)

    model_outputs = np.array([0.97, 0.71, 2.39, 0.97, 2.3, 2.39, 1.87, 2.40, 0.87, 2.15, 1.71, 1.54, 2.15, 2.17, 1.54, 2.2, 1.87, 1.0],
                             dtype=np.float)
    op_vec = model_outputs.reshape(6, 3)
    actual = get_decreased_values(op_vec, up, lo)
    desired = np.array([[0.71, 0.97, 0.87, 2.15, 2.17, 1.87],
                        [0.97, 2.30, 1.87, 1.54, 2.17, 1.87]],
                       dtype=np.float)
    assert_allclose(actual, desired, atol=1e-1)


def test_compute_grouped_mu_star():
    '''
    Computes mu_star for 3 variables grouped into 2 groups
    There are six trajectories.
    '''
    group_matrix = np.matrix('1,0;0,1;0,1', dtype=np.int)
    ee = np.array([[2.52,  2.01,  2.30, -0.66, -0.93, -1.30],
                   [-2.00,  0.13, -0.80,  0.25, -0.02,  0.51],
                   [2.00, -0.13,  0.80, -0.25,  0.02, -0.51]])
    mu_star = np.average(np.abs(ee), 1)
    actual = compute_grouped_mu_star(mu_star, group_matrix)
    desired = np.array([1.62, 0.62], dtype=np.float)
    assert_allclose(actual, desired, rtol=1e-1)
