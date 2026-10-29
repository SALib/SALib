"""Tests to ensure group support in analysis methods (aggregation of indices).

Complements test_groups.py (ProblemSpec-based DataFrame tests) with direct
method-level checks that per-variable indices are averaged per group.
"""

from pytest import fixture

import numpy as np
from numpy.testing import assert_allclose

from SALib.analyze import delta, dgsm, fast, rbd_fast, sobol
from SALib.sample import fast_sampler, finite_diff, latin, saltelli
from SALib.util import compute_groups_matrix, _group_metric

from SALib.test_functions import Ishigami


@fixture(scope="session")
def set_seed():
    """Sets seeds for random generators so that tests can be repeated"""
    seed = 101
    np.random.seed(seed)


@fixture(scope="function")
def test_problem():
    problem = {
        "names": ["x1", "x2", "x3"],
        "num_vars": 3,
        "bounds": [[-np.pi, np.pi]] * 3,
        "groups": ["G1", "G1", "G2"],
    }

    return problem


def test_group_metric_matches_manual_mean():
    """Group aggregate equals the mean of member metric values."""
    groups = ["G1", "G1", "G2"]
    metric = np.array([1.0, 3.0, 10.0])
    groupings, names = compute_groups_matrix(groups)

    actual = _group_metric(groupings, metric)
    assert_allclose(actual, [2.0, 10.0])
    assert list(names) == ["G1", "G2"]


def test_sobol_group_names(set_seed, test_problem):
    X = saltelli.sample(test_problem, 100)
    Y = Ishigami.evaluate(X)
    Si = sobol.analyze(test_problem, Y)

    total, first, second = Si.to_df()
    expected = ["G1", "G2"]

    assert total.index.tolist() == expected
    assert first.index.tolist() == expected
    assert [tuple(idx) for idx in second.index] == [("G1", "G2")]


def test_fast_group(set_seed, test_problem):
    X = fast_sampler.sample(test_problem, 1000, seed=1)
    Y = Ishigami.evaluate(X)
    Si = fast.analyze(test_problem, Y)

    expected = ["G1", "G2"]

    assert Si.to_df().index.tolist() == expected
    assert np.asarray(Si["S1"]).shape[0] == len(expected)


def test_fast_group_matches_mean_of_ungrouped(set_seed, test_problem):
    ungrouped = {
        "names": test_problem["names"],
        "num_vars": 3,
        "bounds": test_problem["bounds"],
    }
    X = fast_sampler.sample(ungrouped, 1000, seed=1)
    Y = Ishigami.evaluate(X)
    Si_flat = fast.analyze(ungrouped, Y)
    Si_grp = fast.analyze(test_problem, Y)

    assert_allclose(Si_grp["S1"][0], np.mean(Si_flat["S1"][:2]))
    assert_allclose(Si_grp["S1"][1], Si_flat["S1"][2])


def test_rbd_fast_group(set_seed, test_problem):
    X = latin.sample(test_problem, 100, seed=1)
    Y = Ishigami.evaluate(X)
    Si = rbd_fast.analyze(test_problem, X, Y)

    expected = ["G1", "G2"]

    assert Si.to_df().index.tolist() == expected
    assert np.asarray(Si["S1"]).shape[0] == len(expected)


def test_delta_group(set_seed, test_problem):
    X = latin.sample(test_problem, 300, seed=1)
    Y = Ishigami.evaluate(X)
    Si = delta.analyze(test_problem, X, Y)

    expected = ["G1", "G2"]

    assert Si.to_df().index.tolist() == expected
    assert np.asarray(Si["delta_balanced"]).shape[0] == len(expected)
    assert len(Si["notes"]) == len(expected)


def test_dgsm_group(set_seed, test_problem):
    X = finite_diff.sample(test_problem, 1000, seed=1)
    Y = Ishigami.evaluate(X)
    Si = dgsm.analyze(test_problem, X, Y)

    expected = ["G1", "G2"]

    assert Si.to_df().index.tolist() == expected
    assert np.asarray(Si["dgsm"]).shape[0] == len(expected)
