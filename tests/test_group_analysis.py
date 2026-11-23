"""Tests to ensure group support in analysis methods (aggregation of indices).

Complements test_groups.py (ProblemSpec-based DataFrame tests) with direct
method-level checks that per-variable indices are aggregated per group.
"""

from pytest import fixture, warns

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


def test_shapley_group(set_seed, test_problem):
    from SALib.analyze import shapley
    from SALib.sample import shapley as shapley_sampler

    with warns(UserWarning, match="group co-sampling"):
        X = shapley_sampler.sample(test_problem, 200, seed=1)
    Y = Ishigami.evaluate(X)
    Si = shapley.analyze(test_problem, X, Y)

    expected = ["G1", "G2"]

    assert Si.to_df().index.tolist() == expected
    assert np.asarray(Si["shapley"]).shape[0] == len(expected)

    # Unlike the other analyzers in this module, Shapley groups are summed,
    # not averaged: the group value equals the sum of the members' values
    # from an ungrouped analysis of the same sample.
    ungrouped_problem = {
        key: value for key, value in test_problem.items() if key != "groups"
    }
    ungrouped = shapley.analyze(ungrouped_problem, X, Y)
    assert_allclose(
        Si["shapley"],
        [ungrouped["shapley"][0] + ungrouped["shapley"][1], ungrouped["shapley"][2]],
    )


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


def test_group_results_ungrouped_snapshot_matches_flat_run(set_seed, test_problem):
    """Si["ungrouped"] holds the pre-aggregation per-factor values.

    For each analyzer that aggregates per-factor metrics, the snapshot
    must equal an ungrouped analysis of the same sample, value for value.
    """
    ungrouped_problem = {
        key: value for key, value in test_problem.items() if key != "groups"
    }

    # FAST
    X = fast_sampler.sample(test_problem, 1000, seed=1)
    Y = Ishigami.evaluate(X)
    Si_grp = fast.analyze(test_problem, Y, seed=1)
    Si_flat = fast.analyze(ungrouped_problem, Y, seed=1)
    assert "ungrouped" in Si_grp
    assert "ungrouped" not in Si_flat
    assert list(Si_grp["ungrouped"]["names"]) == test_problem["names"]
    for key in ("S1", "ST", "S1_conf", "ST_conf"):
        assert_allclose(Si_grp["ungrouped"][key], Si_flat[key])

    # RBD-FAST
    X = latin.sample(test_problem, 100, seed=1)
    Y = Ishigami.evaluate(X)
    Si_grp = rbd_fast.analyze(test_problem, X, Y, seed=1)
    Si_flat = rbd_fast.analyze(ungrouped_problem, X, Y, seed=1)
    for key in ("S1", "S1_conf"):
        assert_allclose(Si_grp["ungrouped"][key], Si_flat[key])

    # DGSM
    X = finite_diff.sample(test_problem, 1000, seed=1)
    Y = Ishigami.evaluate(X)
    Si_grp = dgsm.analyze(test_problem, X, Y, seed=1)
    Si_flat = dgsm.analyze(ungrouped_problem, X, Y, seed=1)
    for key in ("vi", "vi_std", "dgsm", "dgsm_conf"):
        assert_allclose(Si_grp["ungrouped"][key], Si_flat[key])

    # PAWN
    from SALib.analyze import pawn

    X = latin.sample(test_problem, 500, seed=1)
    Y = Ishigami.evaluate(X)
    Si_grp = pawn.analyze(test_problem, X, Y, seed=1)
    Si_flat = pawn.analyze(ungrouped_problem, X, Y, seed=1)
    for key in ("minimum", "mean", "median", "maximum", "CV", "stdev"):
        assert_allclose(Si_grp["ungrouped"][key], Si_flat[key])

    # Discrepancy
    from SALib.analyze import discrepancy

    X = latin.sample(test_problem, 500, seed=1)
    Y = Ishigami.evaluate(X)
    Si_grp = discrepancy.analyze(test_problem, X, Y)
    Si_flat = discrepancy.analyze(ungrouped_problem, X, Y)
    assert_allclose(Si_grp["ungrouped"]["s_discrepancy"], Si_flat["s_discrepancy"])

    # Delta (per-factor notes are kept in the snapshot; the merged
    # per-group notes replace S["notes"] at top level)
    X = latin.sample(test_problem, 300, seed=1)
    Y = Ishigami.evaluate(X)
    Si_grp = delta.analyze(test_problem, X, Y, seed=1)
    Si_flat = delta.analyze(ungrouped_problem, X, Y, seed=1)
    for key in (
        "delta_balanced",
        "delta_balanced_conf",
        "delta_raw",
        "delta_raw_conf",
        "delta_step",
        "delta_step_conf",
        "S1",
        "S1_conf",
    ):
        assert_allclose(Si_grp["ungrouped"][key], Si_flat[key], equal_nan=True)
    assert Si_grp["ungrouped"]["notes"] == Si_flat["notes"]
    assert len(Si_grp["notes"]) == 2  # merged per-group notes

    # Radial EE (sigma keeps the Morris NaN-for-multi-member-groups rule)
    from SALib.analyze import radial_ee
    from SALib.sample.radial.radial_sobol import sample as radial_sample

    X = radial_sample(test_problem, 50, seed=1)
    Y = Ishigami.evaluate(X)
    Si_grp = radial_ee.analyze(test_problem, X, Y, 50, seed=1)
    Si_flat = radial_ee.analyze(ungrouped_problem, X, Y, 50, seed=1)
    for key in ("mu", "mu_star", "mu_star_conf", "sigma"):
        assert_allclose(Si_grp["ungrouped"][key], Si_flat[key])
    assert np.isnan(Si_grp["sigma"][0])
    assert not np.isnan(Si_grp["sigma"][1])


def test_fast_dilution_example_keeps_ungrouped_values(set_seed):
    """Regression test for SALib #708 item 1: group means dilute a single
    dominant member, and the per-factor values are now recoverable from
    Si["ungrouped"]."""
    problem = {
        "num_vars": 4,
        "names": ["x1", "x2", "x3", "x4"],
        "bounds": [[0, 1]] * 4,
        "groups": ["G1", "G1", "G1", "G2"],
    }

    def f(X):
        return 10 * X[:, 0] + 0.0 * X[:, 1] + 0.0 * X[:, 2] + 1.0 * X[:, 3]

    X = fast_sampler.sample(problem, 1024, seed=1)
    Si = fast.analyze(problem, f(X), print_to_console=False, seed=1)

    # The group mean dilutes the dominant member to about a third.
    assert_allclose(Si["S1"][0], 0.3293, atol=1e-4)
    # The per-factor values are preserved in the snapshot.
    assert_allclose(Si["ungrouped"]["S1"][0], 0.9878, atol=1e-4)
    assert_allclose(Si["ungrouped"]["S1"][1:], [0.0, 0.0, 0.0099], atol=1e-4)


def test_group_results_identity_groups_add_no_ungrouped(set_seed):
    """A grouping of one group per factor behaves as no grouping."""
    from SALib.util import ResultDict, group_results

    problem = {
        "num_vars": 3,
        "names": ["x1", "x2", "x3"],
        "bounds": [[-np.pi, np.pi]] * 3,
        "groups": ["x1", "x2", "x3"],
    }
    Si = ResultDict([("S1", np.array([0.5, 0.3, 0.2])), ("names", None)])
    group_results(Si, problem, ["S1"])
    assert "ungrouped" not in Si
    assert list(Si["names"]) == ["x1", "x2", "x3"]
    assert_allclose(Si["S1"], [0.5, 0.3, 0.2])


def test_group_results_sum_aggregation():
    """The "sum" aggregation mode sums members instead of averaging."""
    from SALib.util import ResultDict, group_results

    problem = {
        "num_vars": 3,
        "names": ["x1", "x2", "x3"],
        "bounds": [[0, 1]] * 3,
        "groups": ["G1", "G1", "G2"],
    }
    Si = ResultDict([("S1", np.array([1.0, 3.0, 10.0])), ("names", None)])
    group_results(Si, problem, ["S1"], aggregation="sum")
    assert_allclose(Si["S1"], [4.0, 10.0])
    assert_allclose(Si["ungrouped"]["S1"], [1.0, 3.0, 10.0])


def test_group_results_rejects_unknown_aggregation():
    """An unknown aggregation mode is a loud error, not a silent default."""
    from pytest import raises

    from SALib.util import ResultDict, group_results

    problem = {
        "num_vars": 3,
        "names": ["x1", "x2", "x3"],
        "bounds": [[0, 1]] * 3,
        "groups": ["G1", "G1", "G2"],
    }
    Si = ResultDict([("S1", np.array([1.0, 3.0, 10.0])), ("names", None)])
    with raises(ValueError, match='aggregation must be "mean" or "sum"'):
        group_results(Si, problem, ["S1"], aggregation="median")
    # The check runs before the no-grouping early return, so an invalid
    # mode is loud even when the problem defines no groups.
    ungrouped_problem = {
        key: value for key, value in problem.items() if key != "groups"
    }
    with raises(ValueError, match='aggregation must be "mean" or "sum"'):
        group_results(Si, ungrouped_problem, ["S1"], aggregation="median")


def test_group_results_rejects_unknown_override_key():
    """A typo'd override key must not silently fall back to the mean."""
    from pytest import raises

    from SALib.util import ResultDict, _group_metric, group_results

    problem = {
        "num_vars": 3,
        "names": ["x1", "x2", "x3"],
        "bounds": [[0, 1]] * 3,
        "groups": ["G1", "G1", "G2"],
    }
    Si = ResultDict(
        [
            ("S1", np.array([1.0, 3.0, 10.0])),
            ("sigma", np.array([0.5, 0.5, 0.5])),
            ("names", None),
        ]
    )
    with raises(ValueError, match="Unknown override key"):
        group_results(Si, problem, ["S1", "sigma"], overrides={"Sigma": _group_metric})
    # The check runs before the no-grouping early return, so a typo'd
    # override key is loud even when the problem defines no groups.
    ungrouped_problem = {
        key: value for key, value in problem.items() if key != "groups"
    }
    with raises(ValueError, match="Unknown override key"):
        group_results(
            Si, ungrouped_problem, ["S1", "sigma"], overrides={"Sigma": _group_metric}
        )
    # An override takes the same (groups, values) signature as the
    # built-in aggregators, so passing one of them works as expected.
    group_results(Si, problem, ["S1", "sigma"], overrides={"sigma": _group_metric})
    assert_allclose(Si["sigma"], [0.5, 0.5])


def test_group_results_normalises_metric_keys():
    """A generator of keys must not be exhausted by validation, and a
    bare string must not be read as its characters."""
    from pytest import raises

    from SALib.util import ResultDict, group_results

    problem = {
        "num_vars": 3,
        "names": ["x1", "x2", "x3"],
        "bounds": [[0, 1]] * 3,
        "groups": ["G1", "G1", "G2"],
    }
    Si = ResultDict([("S1", np.array([1.0, 3.0, 10.0])), ("names", None)])
    group_results(Si, problem, (key for key in ["S1"]))
    assert_allclose(Si["S1"], [2.0, 10.0])
    assert_allclose(Si["ungrouped"]["S1"], [1.0, 3.0, 10.0])
    with raises(ValueError, match="not a bare string"):
        group_results(Si, problem, "S1")


def test_group_results_rejects_unknown_metric_key():
    """A typo'd metric key must fail loudly -- including on an ungrouped
    problem, before any mutation happens."""
    from pytest import raises

    from SALib.util import ResultDict, group_results

    problem = {
        "num_vars": 3,
        "names": ["x1", "x2", "x3"],
        "bounds": [[0, 1]] * 3,
        "groups": ["G1", "G1", "G2"],
    }
    ungrouped_problem = {
        key: value for key, value in problem.items() if key != "groups"
    }
    for prob in (problem, ungrouped_problem):
        Si = ResultDict([("S1", np.array([1.0, 3.0, 10.0])), ("names", None)])
        with raises(ValueError, match="Unknown metric key"):
            group_results(Si, prob, ["S1", "S1_confidence"])
        # Nothing was mutated: no snapshot, names untouched.
        assert "ungrouped" not in Si
        assert Si["names"] is None


def test_group_results_to_df_excludes_ungrouped(set_seed, test_problem):
    """Si.to_df() still shows one row per group (no "ungrouped" column)."""
    X = fast_sampler.sample(test_problem, 1000, seed=1)
    Y = Ishigami.evaluate(X)
    Si = fast.analyze(test_problem, Y, seed=1)

    assert Si.to_df().index.tolist() == ["G1", "G2"]
    assert Si.to_df().columns.tolist() == ["S1", "ST", "S1_conf", "ST_conf"]
    # The snapshot itself converts with per-factor names as its index.
    assert Si["ungrouped"].to_df().index.tolist() == ["x1", "x2", "x3"]


def test_grouped_print_to_console_reports_groups(capsys, set_seed, test_problem):
    """print_to_console works on grouped results (no nested key leaks
    into the printed table)."""
    X = fast_sampler.sample(test_problem, 1000, seed=1)
    Y = Ishigami.evaluate(X)
    fast.analyze(test_problem, Y, print_to_console=True, seed=1)
    out = capsys.readouterr().out
    assert "G1" in out and "G2" in out

    X = latin.sample(test_problem, 300, seed=1)
    Y = Ishigami.evaluate(X)
    delta.analyze(test_problem, X, Y, print_to_console=True, seed=1)
    out = capsys.readouterr().out
    assert "G1" in out and "G2" in out
