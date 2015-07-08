from __future__ import division

from nose.tools import with_setup, eq_, raises
from numpy.testing import assert_equal, assert_allclose
import numpy as np

from ..analyze import sobol
from ..sample import saltelli
from ..test_functions import Ishigami, Sobol_G
from ..util import read_param_file


def test_sample_size_second_order():
    param_file = 'SALib/test_functions/params/Ishigami.txt'
    problem = read_param_file(param_file)
    N = 500
    D = 3
    param_values = saltelli.sample(problem, N, calc_second_order=True)
    assert_equal(param_values.shape, [N * (2 * D + 2), D])


def test_sample_size_first_order():
    param_file = 'SALib/test_functions/params/Ishigami.txt'
    problem = read_param_file(param_file)
    N = 500
    D = 3
    param_values = saltelli.sample(problem, N, calc_second_order=False)
    assert_equal(param_values.shape, [N * (D + 2), D])


@raises(RuntimeError)
def test_incorrect_sample_size():
    param_file = 'SALib/test_functions/params/Ishigami.txt'
    problem = read_param_file(param_file)
    param_values = saltelli.sample(problem, 500, calc_second_order=True)
    Y = Ishigami.evaluate(param_values)
    sobol.analyze(problem, Y[:-10], calc_second_order=True)


@raises(RuntimeError)
def test_incorrect_second_order_setting():
    # note this will still be a problem if N(2D+2) also divides by (D+2)
    param_file = 'SALib/test_functions/params/Ishigami.txt'
    problem = read_param_file(param_file)
    param_values = saltelli.sample(problem, 501, calc_second_order=False)
    Y = Ishigami.evaluate(param_values)
    sobol.analyze(problem, Y, calc_second_order=True)


def test_Sobol_G_using_sobol():
    '''
    Tests the accuracy of the Sobol/Saltelli procedure using the Sobol_G
    test function, comparing the results from the Sobol/Saltelli analysis
    against the analytically computed sensitivity index from the Sobol_G
    function.
    '''
    problem = {'num_vars': 6,
               'names': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6'],
               'bounds': [[0, 1], [0, 1], [0, 1], [0, 1],[0, 1], [0, 1]]}
    N = 5000
    a = np.array([78, 12, 0.5, 2, 97, 33])
    param_values = saltelli.sample(problem, N, calc_second_order=False)
    model_results = Sobol_G.evaluate(param_values, a)
    Si = sobol.analyze(problem, model_results, calc_second_order=False)
#     expected = Sobol_G.total_sensitivity_index(a)
#     assert_allclose(Si['ST'], expected)
    expected = Sobol_G.sensitivity_index(a)
    assert_allclose(Si['S1'], expected, atol=1e-2, rtol=1e-6)
