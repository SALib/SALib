from __future__ import division

from itertools import combinations

import numpy as np
import pytest
from pytest import raises

from SALib.analyze import enhanced_hdmr, hdmr
from SALib.sample import latin
from SALib.test_functions import Ishigami, linear_model_1
from SALib.util import read_param_file


def test_enhanced_hdmr_third_order_terms():
    names = ["x1", "x2", "x3", "x4"]
    rng = np.random.default_rng(667)
    X = rng.uniform(-1, 1, size=(300, len(names)))
    Y = X[:, 0] + X[:, 1] * X[:, 2] + X[:, 0] * X[:, 2] * X[:, 3]

    hdmr, _ = enhanced_hdmr._core_params(
        {"names": names.copy()}, len(X), len(names), np.mean(Y), 1, 3, 1, len(X), True
    )
    assert np.isscalar(hdmr.nc3)
    assert hdmr.gamma.shape == (len(list(combinations(names, 3))), 3)

    for max_order in (2, 3):
        expected_terms = names.copy()
        for order in range(2, max_order + 1):
            expected_terms.extend("/".join(term) for term in combinations(names, order))

        result = enhanced_hdmr.analyze(
            {"num_vars": len(names), "names": names.copy()},
            X,
            Y,
            max_order=max_order,
            poly_order=1,
            bootstrap=1,
            seed=667,
        )

        assert result["Term"] == expected_terms
        for key in ("S", "Sa", "Sb", "Signf", "S_conf", "Sa_conf", "Sb_conf"):
            assert result[key].shape == (len(expected_terms),)
            assert np.isfinite(result[key]).all()
        assert result["ST"].shape == (len(expected_terms),)
        assert result["ST_conf"].shape == (len(expected_terms),)
        assert np.isfinite(result["ST"][: len(names)]).all()
        assert np.isfinite(result["ST_conf"][: len(names)]).all()


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
