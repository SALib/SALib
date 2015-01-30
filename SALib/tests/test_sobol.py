from __future__ import division
from numpy.testing import assert_equal, assert_allclose
from nose.tools import with_setup, eq_, raises
from ..sample import saltelli
from ..analyze import sobol
from ..util import read_param_file
from ..test_functions import Ishigami

def test_sample_size_second_order():
    param_file = 'SALib/test_functions/params/Ishigami.txt'
    problem = read_param_file(param_file)
    N = 500
    D = 3
    param_values = saltelli.sample(problem, N, calc_second_order=True)
    assert_equal(param_values.shape, [N*(2*D+2), D])

def test_sample_size_first_order():
    param_file = 'SALib/test_functions/params/Ishigami.txt'
    problem = read_param_file(param_file)
    N = 500
    D = 3
    param_values = saltelli.sample(problem, N, calc_second_order=False)
    assert_equal(param_values.shape, [N*(D+2), D])

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
