from __future__ import division

from numpy.testing import assert_equal

from pytest import raises

import numpy as np

from SALib.sample.morris import sample, _compute_optimised_trajectories
from SALib.util import read_param_file, compute_groups_matrix


def test_group_in_param_file_read(setup_param_file_with_groups):
    '''
    Tests that groups in a parameter file are read correctly
    '''
    parameter_file = setup_param_file_with_groups
    problem = read_param_file(parameter_file)
    groups, group_names = compute_groups_matrix(
        problem['groups'], problem['num_vars'])

    assert_equal(problem['names'], ["Test 1", "Test 2", "Test 3"])
    assert_equal(groups, np.matrix('1,0;1,0;0,1', dtype=np.int))
    assert_equal(group_names, ['Group 1', 'Group 2'])


def test_grid_jump_lt_num_levels(setup_param_file):

    parameter_file = setup_param_file
    problem = read_param_file(parameter_file)

    samples = 10
    num_levels = 4
    grid_jump = 4

    with raises(ValueError):
        sample(problem, samples, num_levels, grid_jump,
               optimal_trajectories=samples)


def test_optimal_trajectories_lt_samples(setup_param_file):

    parameter_file = setup_param_file
    problem = read_param_file(parameter_file)

    samples = 10
    num_levels = 4
    grid_jump = 2

    with raises(ValueError):
        sample(problem, samples, num_levels, grid_jump,
               optimal_trajectories=samples)


def test_optimal_trajectories_lt_10(setup_param_file):

    parameter_file = setup_param_file
    problem = read_param_file(parameter_file)

    samples = 10
    num_levels = 4
    grid_jump = 2
    optimal_trajectories = 11
    with raises(ValueError):
        sample(problem, samples, num_levels, grid_jump,
               optimal_trajectories=optimal_trajectories)


def test_optimal_trajectories_gte_one(setup_param_file):

    parameter_file = setup_param_file
    problem = read_param_file(parameter_file)

    samples = 10
    num_levels = 4
    grid_jump = 2
    optimal_trajectories = 1

    with raises(ValueError):
        sample(problem, samples, num_levels, grid_jump,
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

    output = _compute_optimised_trajectories(
        problem, input_sample, N, k_choices)
    expected = np.concatenate([input_1, input_3, input_4, input_6])
    np.testing.assert_equal(output, expected)


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
    with raises(ValueError):
        _compute_optimised_trajectories(problem, input_sample, N, k_choices)


# @raises(ValueError)
# def test_group_sample_fails_with_no_G_matrix():
#     N = 6
#     num_levels = 4
#     grid_jump = 2
#     problem = {'bounds': [[0., 1.], [0., 1.], [0., 1.], [0., 1.]],
#                'num_vars': 4,
#                'groups': None}
#     sample(problem, N, num_levels, grid_jump)


def test_group_sample_fails_with_wrong_G_matrix():
    N = 6
    num_levels = 4
    grid_jump = 2
    problem = {'bounds': [[0., 1.], [0., 1.], [0., 1.], [0., 1.]],
               'num_vars': 4,
               'groups': (list([1, 2, 3, 4]), None)}
    with raises(TypeError):
        sample(problem, N, num_levels, grid_jump)
