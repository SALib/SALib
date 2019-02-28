from __future__ import division

from nose.tools import with_setup, eq_, raises
from numpy.testing import assert_equal, assert_allclose
import numpy as np
from scipy.stats import norm

from SALib.analyze import sobol
from SALib.sample import saltelli, sobol_sequence
from SALib.test_functions import Ishigami, Sobol_G
from SALib.util import read_param_file


def setup_samples(N = 500, calc_second_order = True):
    param_file = 'src/SALib/test_functions/params/Ishigami.txt'
    problem = read_param_file(param_file)
    param_values = saltelli.sample(problem, N=N, calc_second_order=calc_second_order)
    return problem,param_values


def test_sobol_sequence():
    # example from Joe & Kuo: http://web.maths.unsw.edu.au/~fkuo/sobol/
    S = sobol_sequence.sample(10,3)
    expected = [[0,0,0],[0.5,0.5,0.5],[0.75,0.25,0.25],[0.25,0.75,0.75],
                [0.375,0.375,0.625],[0.875,0.875,0.125],[0.625,0.125,0.875],
                [0.125,0.625,0.375],[0.1875,0.3125,0.9375],[0.6875,0.8125,0.4375]]
    assert_allclose(S, expected, atol=5e-2, rtol=1e-1)


def test_sample_size_second_order():
    N = 500
    D = 3
    problem,param_values = setup_samples(N=N)
    assert_equal(param_values.shape, [N * (2 * D + 2), D])


def test_sample_size_first_order():
    N = 500
    D = 3
    problem,param_values = setup_samples(N=N, calc_second_order=False)
    assert_equal(param_values.shape, [N * (D + 2), D])


@raises(RuntimeError)
def test_incorrect_sample_size():
    problem,param_values = setup_samples()
    Y = Ishigami.evaluate(param_values)
    sobol.analyze(problem, Y[:-10], calc_second_order=True)


@raises(RuntimeError)
def test_bad_conf_level():
    problem,param_values = setup_samples()
    Y = Ishigami.evaluate(param_values)
    Si = sobol.analyze(problem, Y,
                       calc_second_order=True, conf_level=1.01, print_to_console=False)


@raises(RuntimeError)
def test_incorrect_second_order_setting():
    # note this will still be a problem if N(2D+2) also divides by (D+2)
    problem,param_values = setup_samples(N=501, calc_second_order=False)
    Y = Ishigami.evaluate(param_values)
    sobol.analyze(problem, Y, calc_second_order=True)


def test_include_print():
    problem,param_values = setup_samples()
    Y = Ishigami.evaluate(param_values)
    Si = sobol.analyze(problem, Y,
                       calc_second_order=True, conf_level=0.95, print_to_console=True)


def test_parallel_first_order():
    c2o = False
    N = 10000
    problem,param_values = setup_samples(N=N, calc_second_order=c2o)
    Y = Ishigami.evaluate(param_values)

    A,B,AB,BA = sobol.separate_output_values(Y, D=3, N=N,
                                            calc_second_order=c2o)
    r = np.random.randint(N, size=(N, 100))
    Z = norm.ppf(0.5 + 0.95 / 2)
    tasks, n_processors = sobol.create_task_list(D=3,
                            calc_second_order=c2o, n_processors=None)
    Si_list = []
    for t in tasks:
        Si_list.append(sobol.sobol_parallel(Z, A, AB, BA, B, r, t))
    Si = sobol.Si_list_to_dict(Si_list, D=3, calc_second_order=c2o)

    assert_allclose(Si['S1'], [0.31, 0.44, 0.00], atol=5e-2, rtol=1e-1)
    assert_allclose(Si['ST'], [0.55, 0.44, 0.24], atol=5e-2, rtol=1e-1)


def test_parallel_second_order():
    c2o = True
    N = 10000
    problem,param_values = setup_samples(N=N, calc_second_order=c2o)
    Y = Ishigami.evaluate(param_values)

    A,B,AB,BA = sobol.separate_output_values(Y, D=3, N=N,
                                            calc_second_order=c2o)
    r = np.random.randint(N, size=(N, 100))
    Z = norm.ppf(0.5 + 0.95 / 2)
    tasks, n_processors = sobol.create_task_list(D=3,
                            calc_second_order=c2o, n_processors=None)
    Si_list = []
    for t in tasks:
        Si_list.append(sobol.sobol_parallel(Z, A, AB, BA, B, r, t))
    Si = sobol.Si_list_to_dict(Si_list, D=3, calc_second_order=c2o)

    assert_allclose(Si['S1'], [0.31, 0.44, 0.00], atol=5e-2, rtol=1e-1)
    assert_allclose(Si['ST'], [0.55, 0.44, 0.24], atol=5e-2, rtol=1e-1)
    assert_allclose([Si['S2'][0][1], Si['S2'][0][2], Si['S2'][1][2]], [0.00, 0.25, 0.00], atol=5e-2, rtol=1e-1)


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
