from __future__ import division
from numpy.testing import assert_equal, assert_allclose
from nose.tools import with_setup, eq_, raises
from ..sample.morris import Morris
import numpy as np


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


def setup():
    setup_param_file()


@with_setup(setup_param_file_with_groups)
def test_group_in_param_file_read():
    '''
    Tests that groups in a parameter file are read correctly
    '''
    parameter_file = "SALib/tests/test_param_file_w_groups.txt"

    samples = 10
    num_levels = 4
    grid_jump = 2

    sample = Morris(parameter_file, samples, \
                    num_levels, grid_jump, \
                    optimal_trajectories=None)
    assert_equal(sample.parameter_names, ["Test 1", "Test 2", "Test 3"])
    assert_equal(sample.groups, np.matrix('1,0;1,0;0,1', dtype=np.int))
    assert_equal(sample.group_names, ['Group 1', 'Group 2'])


@with_setup(setup)
def test_get_inputs_works():
    '''
    Tests that both the scaled and unscaled methods of the Sample
    class, inherited into the Morris class work correctly.

    This test also checks to ensure that the unscaled sample is
    still correct after computing the scaled sample (which is done
    in place).
    '''
    parameter_file = "SALib/tests/test_param_file.txt"

    samples = 11
    num_levels = 4
    grid_jump = 2

    sample = Morris(parameter_file, samples, num_levels, grid_jump, \
                    optimal_trajectories=None)

    sample.output_sample = np.arange(0,1.1,0.1).repeat(2).reshape((11,2))
    sample.bounds = [[10,20],[-10,10]]

    desired_scaled = np.array([np.arange(10,21,1), np.arange(-10,12,2)],dtype=np.float).T
    desired_unscaled = np.arange(0,1.1,0.1).repeat(2).reshape((11,2))

    actual_unscaled = sample.get_inputs_unscaled()
    assert_allclose(actual_unscaled, desired_unscaled)

    actual_scaled = sample.get_inputs()
    assert_allclose(actual_scaled, desired_scaled)

    actual_unscaled = sample.get_inputs_unscaled()
    assert_allclose(actual_unscaled, desired_unscaled)


@with_setup(setup)
def test_debug():
    '''
    Tests that the debug method of the Morris class does not fail
    when group and optimal_trajectories are equal to None.
    '''
    parameter_file = "SALib/tests/test_param_file.txt"

    samples = 10
    num_levels = 4
    grid_jump = 2

    sample = Morris(parameter_file, samples, num_levels, grid_jump, \
                    optimal_trajectories=None)

    desired = \
    '''Parameter File: SALib/tests/test_param_file.txt\n
    Number of samples: 10\n
    Number of levels: 4\n
    Grid step: 2\n
    Number of variables: 3\n
    Parameter bounds: [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]'''

    #assert_equal(sample.debug(), desired)
    sample.debug()


@with_setup(setup_param_file_with_groups)
def test_compute_groups_from_parameter_file_class():

    samples = 10
    num_levels = 4
    grid_jump = 2

    parameter_file = "SALib/tests/test_param_file_w_groups.txt"

    sample = Morris(parameter_file, samples, num_levels, grid_jump, \
                    optimal_trajectories=None)

    eq_(sample.parameter_file, parameter_file)
    eq_(sample.samples, 10)
    eq_(sample.num_levels, 4)
    eq_(sample.grid_jump, 2)
    eq_(sample.num_vars, 3)
    eq_(sample.bounds, [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    eq_(sample.parameter_names, ["Test 1", "Test 2", "Test 3"])
    assert_equal(sample.groups, np.matrix('1,0;1,0;0,1', dtype=np.int))
    eq_(sample.group_names, ['Group 1', 'Group 2'])
    eq_(sample.optimal_trajectories, None)


@raises(ValueError)
@with_setup(setup, teardown)
def test_optimal_trajectories_lt_samples():

    parameter_file = "SALib/tests/test_param_file.txt"

    samples = 10
    num_levels = 4
    grid_jump = 2

    Morris(parameter_file, samples, num_levels, grid_jump, \
           optimal_trajectories=samples)

@raises(ValueError)
@with_setup(setup, teardown)
def test_optimal_trajectories_lt_10():

    parameter_file = "SALib/tests/test_param_file.txt"

    samples = 10
    num_levels = 4
    grid_jump = 2
    optimal_trajectories = 11

    Morris(parameter_file, samples, num_levels, grid_jump, \
           optimal_trajectories=optimal_trajectories)

@raises(ValueError)
@with_setup(setup, teardown)
def test_optimal_trajectories_gte_one():

    parameter_file = "SALib/tests/test_param_file.txt"

    samples = 10
    num_levels = 4
    grid_jump = 2
    optimal_trajectories = 1

    Morris(parameter_file, samples, num_levels, grid_jump, \
           optimal_trajectories=optimal_trajectories)


@with_setup(setup, teardown)
def test_morris_sample_no_groups_no_optimal_trajectories():

    parameter_file = "SALib/tests/test_param_file.txt"

    samples = 10
    num_levels = 4
    grid_jump = 2


    sample = Morris(parameter_file, samples, num_levels, grid_jump, \
                    optimal_trajectories=None)

    eq_(sample.parameter_file, parameter_file)
    eq_(sample.samples, samples)
    eq_(sample.num_levels, num_levels)
    eq_(sample.grid_jump, grid_jump)
    eq_(sample.num_vars, 3)
    eq_(sample.bounds, [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    eq_(sample.parameter_names, ["Test 1", "Test 2", "Test 3"])
    eq_(sample.groups, None)
    eq_(sample.group_names, None)
    eq_(sample.optimal_trajectories, None)



# Old tests for find_optimal_trajectories function
# This is now inside the Morris class (Morris.optimize_trajectories)
# How to test?

# def test_find_optimum_trajectories():
#     input_1 = [[0, 1/3.], [0, 1.], [2/3., 1.]]
#     input_2 = [[0, 1/3.], [2/3., 1/3.], [2/3., 1.]]
#     input_3 = [[2/3., 0], [2/3., 2/3.], [0, 2/3.]]
#     input_4 = [[1/3., 1.], [1., 1.], [1, 1/3.]]
#     input_5 = [[1/3., 1.], [1/3., 1/3.], [1, 1/3.]]
#     input_6 = [[1/3., 2/3.], [1/3., 0], [1., 0]]
#     input_sample = np.concatenate([input_1, input_2, input_3,
#                                    input_4, input_5, input_6])
#     N = 6
#     num_params = 2
#     k_choices = 4
#     output = find_optimum_trajectories(input_sample, N, num_params, k_choices)
#     expected = np.concatenate([input_1, input_3, input_4, input_6])
#     np.testing.assert_equal(output, expected)


# @raises(ValueError)
# def test_catch_inputs_not_in_zero_one_range():
#     num_params = 2
#     k_choices = 4
#     N = 10
#     input_sample = setup() * 10
#     find_optimum_trajectories(input_sample, N, num_params, k_choices)
