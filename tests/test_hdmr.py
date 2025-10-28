from __future__ import division

import pytest
from pytest import raises

from SALib.analyze import hdmr
from SALib.sample import latin
from SALib.test_functions import Ishigami, linear_model_1
from SALib.util import read_param_file


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def setup_samples(N=10000):
    param_file = "src/SALib/test_functions/params/Ishigami.txt"
    problem = read_param_file(param_file)
    param_values = latin.sample(problem, 10000)
    return problem, param_values


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_insufficient_sample_size():
    problem, X = setup_samples()
    Y = Ishigami.evaluate(X)
    with raises(RuntimeError):
        hdmr.analyze(problem, X[:200], Y[:200])


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_bad_conf_level():
    problem, X = setup_samples()
    Y = Ishigami.evaluate(X)
    with raises(RuntimeError):
        hdmr.analyze(problem, X, Y, alpha=1.02)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_incorrect_maxorder():
    problem, X = setup_samples()
    Y = Ishigami.evaluate(X)
    with raises(RuntimeError):
        hdmr.analyze(problem, X, Y, maxorder=4)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_incorrect_maxiter():
    problem, X = setup_samples()
    Y = Ishigami.evaluate(X)
    with raises(RuntimeError):
        hdmr.analyze(problem, X, Y, maxiter=1005)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_over_bootstrap_sample_size():
    problem, X = setup_samples()
    Y = Ishigami.evaluate(X)
    with raises(RuntimeError):
        hdmr.analyze(problem, X, Y, R=10001)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_incorrect_maxorder_setting():
    problem = {"num_vars": 2, "names": ["x1", "x2"], "bounds": [[0, 1] * 2]}
    X = latin.sample(problem, 10000)
    Y = linear_model_1.evaluate(X)
    with raises(RuntimeError):
        hdmr.analyze(problem, X, Y, maxorder=5)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_incorrect_lambdax():
    problem, X = setup_samples()
    Y = Ishigami.evaluate(X)
    with raises(RuntimeError):
        hdmr.analyze(problem, X, Y, lambdax=11)


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_dim_mismatch():
    problem = {"num_vars": 2, "names": ["x1", "x2"], "bounds": [[0, 1] * 2]}
    X = latin.sample(problem, 10000)
    Y = linear_model_1.evaluate(X)
    with raises(RuntimeError):
        hdmr.analyze(problem, X, Y[:-2])
