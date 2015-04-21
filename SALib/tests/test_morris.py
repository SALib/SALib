from __future__ import division

from nose.tools import with_setup, eq_, raises
from numpy.testing import assert_equal, assert_allclose

import numpy as np

from ..sample.morris import sample, compute_optimised_trajectories
from ..util import read_param_file


def teardown():
    pass


def setup_param_file():
    filename = "SALib/tests/test_param_file.txt"
    with open(filename, "w") as ofile:
        ofile.write("Test 1,0,1.0\n")
        ofile.write("Test 2,0,1.0\n")
        ofile.write("Test 3,0,1.0\n")


def setup_param_file_with_groups():
    filename = "SALib/tests/test_param_file_w_groups.txt"
    with open(filename, "w") as ofile:
        ofile.write("Test 1,0,1.0,Group 1\n")
        ofile.write("Test 2,0,1.0,Group 1\n")
        ofile.write("Test 3,0,1.0,Group 2\n")


def setup_param_file_with_groups_prime():
    filename = "SALib/tests/test_param_file_w_groups_prime.txt"
    with open(filename, "w") as ofile:
        ofile.write("Test 1,0,1.0,Group 1\n")
        ofile.write("Test 2,0,1.0,Group 2\n")
        ofile.write("Test 3,0,1.0,Group 2\n")
        ofile.write("Test 4,0,1.0,Group 3\n")
        ofile.write("Test 5,0,1.0,Group 3\n")
        ofile.write("Test 6,0,1.0,Group 3\n")
        ofile.write("Test 7,0,1.0,Group 3\n")


def setup():
    setup_param_file()
    setup_param_file_with_groups()
    setup_param_file_with_groups_prime()


@with_setup(setup_param_file_with_groups)
def test_group_in_param_file_read():
    '''
    Tests that groups in a parameter file are read correctly
    '''
    parameter_file = "SALib/tests/test_param_file_w_groups.txt"
    problem = read_param_file(parameter_file)
    groups, group_names = problem['groups']

    assert_equal(problem['names'], ["Test 1", "Test 2", "Test 3"])
    assert_equal(groups, np.matrix('1,0;1,0;0,1', dtype=np.int))
    assert_equal(group_names, ['Group 1', 'Group 2'])


@raises(ValueError)
@with_setup(setup, teardown)
def test_grid_jump_lt_num_levels():

    parameter_file = "SALib/tests/test_param_file.txt"
    problem = read_param_file(parameter_file)

    samples = 10
    num_levels = 4
    grid_jump = 4

    sample(problem, samples, num_levels, grid_jump, \
           optimal_trajectories=samples)


@raises(ValueError)
@with_setup(setup, teardown)
def test_optimal_trajectories_lt_samples():

    parameter_file = "SALib/tests/test_param_file.txt"
    problem = read_param_file(parameter_file)

    samples = 10
    num_levels = 4
    grid_jump = 2

    sample(problem, samples, num_levels, grid_jump, \
           optimal_trajectories=samples)


@raises(ValueError)
@with_setup(setup, teardown)
def test_optimal_trajectories_lt_10():

    parameter_file = "SALib/tests/test_param_file.txt"
    problem = read_param_file(parameter_file)

    samples = 10
    num_levels = 4
    grid_jump = 2
    optimal_trajectories = 11

    sample(problem, samples, num_levels, grid_jump, \
           optimal_trajectories=optimal_trajectories)

@raises(ValueError)
@with_setup(setup, teardown)
def test_optimal_trajectories_gte_one():

    parameter_file = "SALib/tests/test_param_file.txt"
    problem = read_param_file(parameter_file)

    samples = 10
    num_levels = 4
    grid_jump = 2
    optimal_trajectories = 1

    sample(problem, samples, num_levels, grid_jump, \
           optimal_trajectories=optimal_trajectories)


def test_find_optimum_trajectories():
    input_1 = [[0, 1 / 3.], [0, 1.], [2 / 3., 1.]]
    input_2 = [[0, 1 / 3.], [2 / 3., 1 / 3.], [2 / 3., 1.]]
    input_3 = [[2 / 3., 0], [2 / 3., 2 / 3.], [0, 2 / 3.]]
    input_4 = [[1 / 3., 1.], [1., 1.], [1, 1 / 3.]]
    input_5 = [[1 / 3., 1.], [1 / 3., 1 / 3.], [1, 1 / 3.]]
    input_6 = [[1 / 3., 2 / 3.], [1 / 3., 0], [1., 0]]
    input_sample = np.concatenate([input_1, input_2, input_3,
                                   input_4, input_5, input_6])
    N = 6
    problem = {'num_vars': 2, 'groups': None}
    k_choices = 4

    output = compute_optimised_trajectories(problem, input_sample, N, k_choices)
    expected = np.concatenate([input_1, input_3, input_4, input_6])
    np.testing.assert_equal(output, expected)


@raises(ValueError)
def test_catch_inputs_not_in_zero_one_range():
    input_1 = [[0, 1 / 3.], [0, 1.], [2 / 3., 1.]]
    input_2 = [[0, 1 / 3.], [2 / 3., 1 / 3.], [2 / 3., 1.]]
    input_3 = [[2 / 3., 0], [2 / 3., 2 / 3.], [0, 2 / 3.]]
    input_4 = [[1 / 3., 1.], [1., 1.], [1, 1 / 3.]]
    input_5 = [[1 / 3., 1.], [1 / 3., 1 / 3.], [1, 1 / 3.]]
    input_6 = [[1 / 3., 2 / 3.], [1 / 3., 0], [1., 0]]
    input_sample = np.concatenate([input_1, input_2, input_3,
                                   input_4, input_5, input_6])
    problem = {'num_vars': 2, 'groups': None}
    k_choices = 4
    N = 10
    input_sample *= 10
    compute_optimised_trajectories(problem, input_sample, N, k_choices)

@raises(ValueError)
def test_group_sample_fails_with_no_G_matrix():
    N = 6
    num_levels = 4
    grid_jump = 2
    problem = {'bounds': [[0., 1.], [0., 1.], [0., 1.], [0., 1.]],
               'num_vars': 4,
               'groups': (None, None)}
    sample(problem, N, num_levels, grid_jump)


@raises(TypeError)
def test_group_sample_fails_with_wrong_G_matrix():
    N = 6
    num_levels = 4
    grid_jump = 2
    problem = {'bounds': [[0., 1.], [0., 1.], [0., 1.], [0., 1.]],
               'num_vars': 4,
               'groups': (list[1, 2, 3, 4], None)}
    sample(problem, N, num_levels, grid_jump)
