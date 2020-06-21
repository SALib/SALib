from __future__ import division

from pytest import raises
import numpy as np

from SALib.analyze import hdmr
from SALib.sample import latin
from SALib.test_functions import Ishigami, linear_model_1
from SALib.util import read_param_file


def setup_samples(N=10000):
    param_file = 'src/SALib/test_functions/params/Ishigami.txt'
    problem = read_param_file(param_file)
    param_values = latin.sample(problem, 10000)
    return problem, param_values


def test_insufficient_sample_size():
    problem, X = setup_samples()
    Y = Ishigami.evaluate(X)
    with raises(RuntimeError):
        hdmr.analyze(problem, X[:200], Y[:200])


def test_bad_conf_level():
    problem, X = setup_samples()
    Y = Ishigami.evaluate(X)
    with raises(RuntimeError):
        hdmr.analyze(problem, X, Y, alpha=1.02)


def test_incorrect_maxorder():
    problem, X = setup_samples()
    Y = Ishigami.evaluate(X)
    with raises(RuntimeError):
        hdmr.analyze(problem, X, Y, maxorder=4)


def test_incorrect_maxiter():
    problem, X = setup_samples()
    Y = Ishigami.evaluate(X)
    with raises(RuntimeError):
        hdmr.analyze(problem, X, Y, maxiter=1005)


def test_over_bootstrap_sample_size():
    problem, X = setup_samples()
    Y = Ishigami.evaluate(X)
    with raises(RuntimeError):
        hdmr.analyze(problem, X, Y, R=10001)


def test_incorrect_maxorder_setting():
    problem = {
        'num_vars': 2,
        'names': ['x1', 'x2'],
        'bounds': [[0, 1]*2]
    }
    X = latin.sample(problem, 10000)
    Y = linear_model_1.evaluate(X)
    with raises(RuntimeError):
        hdmr.analyze(problem, X, Y, maxorder=5)


def test_incorrect_lambdax():
    problem, X = setup_samples()
    Y = Ishigami.evaluate(X)
    with raises(RuntimeError):
        hdmr.analyze(problem, X, Y, lambdax=11)


def test_dim_mismatch():
    problem = {
        'num_vars': 2,
        'names': ['x1', 'x2'],
        'bounds': [[0, 1]*2]
    }
    X = latin.sample(problem, 10000)
    Y = linear_model_1.evaluate(X)
    with raises(RuntimeError):
        hdmr.analyze(problem, X, Y[:-2])