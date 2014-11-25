from __future__ import division
from numpy.testing import assert_equal, assert_allclose
from nose.tools import raises, with_setup
from ..sample.morris import Morris
import numpy as np


def test_get_input_sample_unscaled():
    pass

def setup_param_file():
    filename = "SALib/tests/test_param_file.txt"
    with open(filename, "w") as ofile:
         ofile.write("Test 1,0,1.0\n")
         ofile.write("Test 2,0,1.0\n")
         ofile.write("Test 3,0,1.0\n")


def setup_group_file():
    filename = "SALib/tests/test_group_file.txt"
    with open(filename, "w") as ofile:
         ofile.write("Test 1,1,0\n")
         ofile.write("Test 2,0,1\n")
         ofile.write("Test 3,0,1\n")


def setup():
    setup_param_file()
    setup_group_file()


@with_setup(setup)
def test_group_file_read():
    '''
    Tests that a group file is read correctly
    '''
    parameter_file = "SALib/tests/test_param_file.txt"
    group_file = "SALib/tests/test_group_file.txt"

    samples = 10
    num_levels = 4
    grid_jump = 2

    sample = Morris(parameter_file, samples, num_levels, grid_jump, \
                 group=group_file, optimal_trajectories=None)

    assert_equal(sample.parameter_names, ["Test 1", "Test 2", "Test 3"])
    assert_equal(sample.groups, np.matrix('1,0;0,1;0,1'))


@with_setup(setup)
def test_get_input_sample_scaled_works():
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
                 group=None, optimal_trajectories=None)

    sample.output_sample = np.arange(0,1.1,0.1).repeat(2).reshape((11,2))
    sample.bounds = [[10,20],[-10,10]]

    desired_scaled = np.array([np.arange(10,21,1), np.arange(-10,12,2)],dtype=np.float).T
    desired_unscaled = np.arange(0,1.1,0.1).repeat(2).reshape((11,2))

    actual_unscaled = sample.get_input_sample_unscaled()
    assert_allclose(actual_unscaled, desired_unscaled)

    actual_scaled = sample.get_input_sample_scaled()
    assert_allclose(actual_scaled, desired_scaled)

    actual_unscaled = sample.get_input_sample_unscaled()
    assert_allclose(actual_unscaled, desired_unscaled)
