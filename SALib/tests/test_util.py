from __future__ import division
from numpy.testing import assert_equal
from nose.tools import raises, with_setup
from ..util import read_param_file, scale_samples
import os


def setup_function():
    filename = "SALib/tests/test_params.txt"
    with open(filename, "w") as ofile:
         ofile.write("Test1 0.0 100.0\n")
         ofile.write("Test2 5.0 51.0\n")


def setup_csv_param_file_with_whitespace_in_names():
    filename = "SALib/tests/test_params_csv_whitespace.txt"
    with open(filename, "w") as ofile:
         ofile.write("Test 1,0.0,100.0\n")
         ofile.write("Test 2,5.0,51.0\n")


def setup_tab_param_file_with_whitespace_in_names():
    filename = "SALib/tests/test_params_tab_whitespace.txt"
    with open(filename, "w") as ofile:
         ofile.write("Test 1\t0.0\t100.0\n")
         ofile.write("Test 2\t5.0\t51.0\n")

def teardown():
    [os.remove("SALib/tests/%s" % f) for f in os.listdir("SALib/tests/") if f.endswith(".txt")]


@with_setup(setup_function, teardown)
def test_readfile():
    '''
    Tests a standard parameter file is read correctly
    '''

    filename = "SALib/tests/test_params.txt"
    pf = read_param_file(filename)

    assert_equal(pf['bounds'], [[0, 100], [5, 51]])
    assert_equal(pf['num_vars'], 2)
    assert_equal(pf['names'], ['Test1', 'Test2'])


@with_setup(setup_csv_param_file_with_whitespace_in_names, teardown)
def test_csv_readfile_with_whitespace():
    '''
    A comma delimited parameter file with whitespace in the names
    '''

    filename = "SALib/tests/test_params_csv_whitespace.txt"
    pf = read_param_file(filename)

    assert_equal(pf['bounds'], [[0, 100], [5, 51]])
    assert_equal(pf['num_vars'], 2)
    assert_equal(pf['names'], ['Test 1', 'Test 2'])


@with_setup(setup_tab_param_file_with_whitespace_in_names, teardown)
def test_tab_readfile_with_whitespace():
    '''
    A tab delimited parameter file with whitespace in the names
    '''

    filename = "SALib/tests/test_params_tab_whitespace.txt"
    pf = read_param_file(filename)

    assert_equal(pf['bounds'], [[0, 100], [5, 51]])
    assert_equal(pf['num_vars'], 2)
    assert_equal(pf['names'], ['Test 1', 'Test 2'])


def test_scale_samples():
    pass
