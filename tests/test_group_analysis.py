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


def test_pawn_group(set_seed, test_problem):
    from SALib.analyze import pawn

    X = latin.sample(test_problem, 500, seed=1)
    Y = Ishigami.evaluate(X)
    Si = pawn.analyze(test_problem, X, Y, seed=1)

    expected = ["G1", "G2"]

    assert Si.to_df().index.tolist() == expected
    for key in ("minimum", "mean", "median", "maximum", "CV", "stdev"):
        assert np.asarray(Si[key]).shape[0] == len(expected)


def test_discrepancy_group(set_seed, test_problem):
    from SALib.analyze import discrepancy

    X = latin.sample(test_problem, 500, seed=1)
    Y = Ishigami.evaluate(X)
    Si = discrepancy.analyze(test_problem, X, Y)

    expected = ["G1", "G2"]

    assert Si["names"] == expected
    assert Si.to_df().index.tolist() == expected
    assert np.asarray(Si["s_discrepancy"]).shape[0] == len(expected)


def test_discrepancy_group_matches_mean_of_ungrouped(set_seed, test_problem):
    from SALib.analyze import discrepancy

    ungrouped = {
        "names": test_problem["names"],
        "num_vars": 3,
        "bounds": test_problem["bounds"],
    }
    X = latin.sample(ungrouped, 500, seed=1)
    Y = Ishigami.evaluate(X)
    Si_flat = discrepancy.analyze(ungrouped, X, Y)
    Si_grp = discrepancy.analyze(test_problem, X, Y)

    assert_allclose(Si_grp["s_discrepancy"][0], np.mean(Si_flat["s_discrepancy"][:2]))
    assert_allclose(Si_grp["s_discrepancy"][1], Si_flat["s_discrepancy"][2])


def test_rsa_group(set_seed, test_problem):
    from SALib.analyze import rsa

    X = latin.sample(test_problem, 500, seed=1)
    Y = Ishigami.evaluate(X)
    Si = rsa.analyze(test_problem, X, Y)

    expected = ["G1", "G2"]

    assert [str(n) for n in Si["names"]] == expected
    assert Si.to_df().columns.tolist() == expected


def test_radial_ee_group(set_seed, test_problem):
    from SALib.analyze import radial_ee
    from SALib.sample.radial.radial_sobol import sample as radial_sample

    sample_sets = 50
    X = radial_sample(test_problem, sample_sets, seed=1)
    Y = Ishigami.evaluate(X)
    Si = radial_ee.analyze(test_problem, X, Y, sample_sets, seed=1)

    expected = ["G1", "G2"]

    assert [str(n) for n in Si["names"]] == expected
    for key in ("mu", "mu_star", "mu_star_conf", "sigma"):
        assert np.asarray(Si[key]).shape[0] == len(expected)
    # sigma is NaN for multi-member groups (Morris convention)
    assert np.isnan(Si["sigma"][0])
    assert not np.isnan(Si["sigma"][1])


def test_single_group_reported_as_one_group(set_seed):
    """A problem defining a single group reports one aggregated row, not
    individual factors."""
    from SALib.analyze import discrepancy, pawn, rsa

    problem = {
        "names": ["x1", "x2", "x3"],
        "num_vars": 3,
        "bounds": [[-np.pi, np.pi]] * 3,
        "groups": ["G1", "G1", "G1"],
    }

    X = latin.sample(problem, 500, seed=1)
    Y = Ishigami.evaluate(X)

    Si = pawn.analyze(problem, X, Y, seed=1)
    assert [str(n) for n in Si["names"]] == ["G1"]
    for key in ("minimum", "mean", "median", "maximum", "CV", "stdev"):
        assert np.asarray(Si[key]).shape[0] == 1

    Si = discrepancy.analyze(problem, X, Y)
    assert [str(n) for n in Si["names"]] == ["G1"]
    assert np.asarray(Si["s_discrepancy"]).shape[0] == 1

    Si = rsa.analyze(problem, X, Y)
    assert [str(n) for n in Si["names"]] == ["G1"]


def test_single_group_dgsm_fast_rbd(set_seed):
    """Ported analyzers also report a single group as one aggregated row."""
    problem = {
        "names": ["x1", "x2", "x3"],
        "num_vars": 3,
        "bounds": [[-np.pi, np.pi]] * 3,
        "groups": ["G1", "G1", "G1"],
    }

    X = finite_diff.sample(problem, 100, seed=1)
    Y = Ishigami.evaluate(X)
    Si = dgsm.analyze(problem, X, Y)
    assert [str(n) for n in Si["names"]] == ["G1"]
    assert np.asarray(Si["dgsm"]).shape[0] == 1

    X = latin.sample(problem, 300, seed=1)
    Y = Ishigami.evaluate(X)
    Si = rbd_fast.analyze(problem, X, Y)
    assert [str(n) for n in Si["names"]] == ["G1"]
    assert np.asarray(Si["S1"]).shape[0] == 1
