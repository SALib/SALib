"""Regression tests for the enhanced_hdmr analyzer.

Covers https://github.com/SALib/SALib/issues/642, where ``analyze()`` with
``max_order=1`` raised ``AttributeError: 'CoreParams' object has no
attribute 'S'`` inside ``_finalize``, and the related third-order indexing
bug which read ``hdmr.nc3`` (an integer count) instead of ``hdmr.gamma``
(the array of three-factor combinations).
"""

import numpy as np
from numpy.testing import assert_allclose, assert_array_less

from SALib.analyze import enhanced_hdmr
from SALib.sample import latin


def _sample(problem, seed=101, n=300):
    X = latin.sample(problem, n, seed=seed)
    y = np.random.default_rng(seed).standard_normal(len(X))
    return X, y


def test_enhanced_hdmr_max_order_one():
    """max_order=1 should not raise and ST should equal S (#642)."""
    problem = {
        "num_vars": 4,
        "names": ["x_1", "x_2", "x_3", "x_4"],
        "bounds": [[0.0, 1.0]] * 4,
    }
    X, y = _sample(problem)

    Si = enhanced_hdmr.analyze(
        problem,
        X,
        y,
        max_order=1,
        max_iter=10,
        alpha=0.95,
        print_to_console=False,
        seed=101,
    )

    # With first-order terms only, total sensitivity equals first-order
    # sensitivity for every factor.
    assert_allclose(Si["ST"], Si["S"][: problem["num_vars"]])
    assert_allclose(Si["ST_conf"], Si["S_conf"][: problem["num_vars"]])


def test_enhanced_hdmr_max_order_three():
    """max_order=3 should not raise and ST should be >= first-order S."""
    problem = {
        "num_vars": 4,
        "names": ["x_1", "x_2", "x_3", "x_4"],
        "bounds": [[0.0, 1.0]] * 4,
    }
    X, y = _sample(problem)

    Si = enhanced_hdmr.analyze(
        problem,
        X,
        y,
        max_order=3,
        max_iter=10,
        alpha=0.95,
        print_to_console=False,
        seed=101,
    )

    d = problem["num_vars"]
    assert np.all(np.isfinite(Si["ST"][:d]))
    # Total sensitivity of a factor must be at least its first-order index
    assert_array_less(Si["S"][:d], Si["ST"][:d] + 1e-12)


def test_enhanced_hdmr_max_order_two():
    """Sanity check that the previously working second-order path is intact."""
    problem = {
        "num_vars": 3,
        "names": ["x_1", "x_2", "x_3"],
        "bounds": [[0.0, 1.0]] * 3,
    }
    X, y = _sample(problem)

    Si = enhanced_hdmr.analyze(
        problem,
        X,
        y,
        max_order=2,
        max_iter=10,
        alpha=0.95,
        print_to_console=False,
        seed=101,
    )

    d = problem["num_vars"]
    assert np.all(np.isfinite(Si["ST"][:d]))
    assert_array_less(Si["S"][:d], Si["ST"][:d] + 1e-12)
