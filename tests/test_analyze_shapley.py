import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm

from SALib import ProblemSpec
from SALib.analyze import shapley
from SALib.sample import shapley as shapley_sampler
from SALib.test_functions import Ishigami


@pytest.fixture
def two_factor_problem():
    return {
        "num_vars": 2,
        "names": ["x1", "x2"],
        "bounds": [[0.0, 1.0], [0.0, 1.0]],
    }


def test_analyze_matches_goda_update_and_variance_formula(two_factor_problem):
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, 2.0],
            [3.0, 4.0],
            [2.0, 1.0],
            [2.0, 5.0],
            [4.0, 5.0],
        ]
    )
    Y = X[:, 0] * X[:, 1]

    result = shapley.analyze(two_factor_problem, X, Y)

    assert_allclose(result["shapley"], [69.0, 37.0])
    assert_allclose(
        result["shapley_conf"],
        norm.ppf(0.975) * np.array([61.0, 5.0]),
    )
    assert_allclose(result["shapley_normalized"], [69.0 / 106.0, 37.0 / 106.0])
    assert_allclose(result["shapley_normalized"].sum(), 1.0)
    assert result["names"] == ["x1", "x2"]

    # Delta-method half-width for a ratio of correlated trajectory means;
    # both factors land on the same value here since there are only two
    # trajectories, so their contributions are mirror images about the mean.
    expected_half_width = norm.ppf(0.975) * (1301.0 / 5618.0)
    assert_allclose(
        result["shapley_normalized_conf"], [expected_half_width, expected_half_width]
    )


def test_plot_defaults_to_normalized(two_factor_problem):
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, 2.0],
            [3.0, 4.0],
            [2.0, 1.0],
            [2.0, 5.0],
            [4.0, 5.0],
        ]
    )
    Y = X[:, 0] * X[:, 1]

    result = shapley.analyze(two_factor_problem, X, Y)

    # Raw (variance-unit) and normalized (unit-interval) effects sit on
    # incompatible scales, so plot() shouldn't mix them in one chart (the
    # behavior ResultDict.plot() would otherwise give); default to the
    # normalized shares, since those are what's usually of interest.
    ax = result.plot()
    assert ax.get_title() == "Normalized Shapley effects"

    ax = result.plot(normalized=False)
    assert ax.get_title() == "Shapley effects"


def test_normalized_effects_reject_zero_total(two_factor_problem):
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 2.0],
            [3.0, 3.0],
        ]
    )
    result = shapley.analyze(two_factor_problem, X, np.ones(6))

    # Normalization is undefined when the estimated total is zero; degrades
    # to NaN rather than raising, consistent with SALib's degenerate-case
    # handling elsewhere (e.g. Sobol's zero-variance guard).
    assert np.all(np.isnan(result["shapley_normalized"]))
    assert np.all(np.isnan(result["shapley_normalized_conf"]))


def test_ishigami_regression():
    problem = {
        "num_vars": 3,
        "names": ["x1", "x2", "x3"],
        "bounds": [[-np.pi, np.pi]] * 3,
    }
    X = shapley_sampler.sample(problem, 20_000, seed=1234)
    Y = Ishigami.evaluate(X)

    result = shapley.analyze(problem, X, Y)

    a, b = 7.0, 0.1
    interaction_share = 4 * np.pi**8 * b**2 / 225
    expected = np.array(
        [
            0.5 + b * np.pi**4 / 5 + b**2 * np.pi**8 / 50 + interaction_share,
            a**2 / 8,
            interaction_share,
        ]
    )
    assert_allclose(result["shapley"], expected, atol=0.18)

    trajectory_outputs = Y.reshape(20_000, 4)
    variance_estimate = np.mean(
        (trajectory_outputs[:, 0] - trajectory_outputs[:, -1]) ** 2 / 2
    )
    assert_allclose(np.sum(result["shapley"]), variance_estimate)


def test_problem_spec_interface():
    specification = ProblemSpec(
        {
            "names": ["x1", "x2", "x3"],
            "bounds": [[-np.pi, np.pi]] * 3,
            "outputs": ["Y"],
        }
    )

    (
        specification.sample_shapley(256, seed=123)
        .evaluate(Ishigami.evaluate)
        .analyze_shapley()
    )

    assert specification.analysis["names"] == ["x1", "x2", "x3"]

    # Normalized shares and their confidence bounds are plain result keys, so
    # they surface through ProblemSpec.to_df() with no special-cased access.
    df = specification.to_df()
    assert "shapley_normalized" in df.columns
    assert "shapley_normalized_conf" in df.columns
    assert_allclose(df["shapley_normalized"].sum(), 1.0)


@pytest.mark.parametrize("conf_level", [0.0, 1.0, -0.1, 1.1])
def test_analyze_rejects_invalid_confidence(two_factor_problem, conf_level):
    X = np.zeros((6, 2))
    Y = np.zeros(6)

    with pytest.raises(ValueError, match="between 0 and 1"):
        shapley.analyze(two_factor_problem, X, Y, conf_level=conf_level)


def test_analyze_rejects_malformed_trajectory(two_factor_problem):
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [3.0, 2.0],
            [3.0, 4.0],
        ]
    )
    Y = X.sum(axis=1)

    with pytest.raises(ValueError, match="exactly one factor"):
        shapley.analyze(two_factor_problem, X, Y)


def test_analyze_aggregates_groups_by_summation():
    problem = {
        "num_vars": 3,
        "names": ["x1", "x2", "x3"],
        "bounds": [[-np.pi, np.pi]] * 3,
        "groups": ["G1", "G1", "G2"],
    }
    with pytest.warns(UserWarning, match="group co-sampling"):
        X = shapley_sampler.sample(problem, 500, seed=123)
    Y = Ishigami.evaluate(X)

    grouped = shapley.analyze(problem, X, Y)
    ungrouped_problem = {
        key: value for key, value in problem.items() if key != "groups"
    }
    ungrouped = shapley.analyze(ungrouped_problem, X, Y)

    assert list(grouped["names"]) == ["G1", "G2"]
    # Bundle property: a group's effect is the sum of its members' effects,
    # not their average, so the total-variance decomposition is preserved.
    assert_allclose(
        grouped["shapley"],
        [
            ungrouped["shapley"][0] + ungrouped["shapley"][1],
            ungrouped["shapley"][2],
        ],
    )
    assert_allclose(grouped["shapley"].sum(), ungrouped["shapley"].sum())
    assert_allclose(grouped["shapley_normalized"].sum(), 1.0)


def test_analyze_single_group_collapses_to_one_row(two_factor_problem):
    two_factor_problem["groups"] = ["G", "G"]
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, 2.0],
            [3.0, 4.0],
            [2.0, 1.0],
            [2.0, 5.0],
            [4.0, 5.0],
        ]
    )
    Y = X[:, 0] * X[:, 1]

    result = shapley.analyze(two_factor_problem, X, Y)

    assert list(result["names"]) == ["G"]
    assert_allclose(result["shapley"], [106.0])
    assert_allclose(result["shapley_normalized"], [1.0])


def test_analyze_identity_grouping_matches_ungrouped(two_factor_problem):
    X = np.array(
        [
            [1.0, 2.0],
            [3.0, 2.0],
            [3.0, 4.0],
            [2.0, 1.0],
            [2.0, 5.0],
            [4.0, 5.0],
        ]
    )
    Y = X[:, 0] * X[:, 1]

    ungrouped = shapley.analyze(two_factor_problem, X, Y)
    two_factor_problem["groups"] = ["x1", "x2"]
    grouped = shapley.analyze(two_factor_problem, X, Y)

    for key in (
        "shapley",
        "shapley_conf",
        "shapley_abs",
        "shapley_abs_conf",
        "shapley_normalized",
        "shapley_normalized_conf",
    ):
        assert_allclose(grouped[key], ungrouped[key])
    assert list(grouped["names"]) == list(ungrouped["names"])


def test_shapley_abs_reveals_cancelling_contributions(two_factor_problem):
    # Hand-computed contributions: trajectory 1 gives (0.5, 4.0) and
    # trajectory 2 gives (-4.0, 4.5), so the signed mean for x1 understates
    # its per-trajectory contributions, which the absolute mean exposes --
    # the same signed/absolute pairing as Morris' mu and mu_star.
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    Y = np.array([0.0, 1.0, 3.0, 0.0, 3.0, 1.0])

    result = shapley.analyze(two_factor_problem, X, Y)

    assert_allclose(result["shapley"], [-1.75, 4.25])
    assert_allclose(result["shapley_abs"], [2.25, 4.25])


@pytest.mark.parametrize(
    ("X", "Y", "message"),
    [
        (np.zeros(6), np.zeros(6), "X must have shape"),
        (np.zeros((6, 3)), np.zeros(6), "X must have shape"),
        (np.zeros((6, 2)), np.zeros((6, 1)), "one-dimensional"),
        (np.zeros((6, 2)), np.zeros(5), "same number of rows"),
        (np.zeros((5, 2)), np.zeros(5), "divisible"),
        (np.zeros((3, 2)), np.zeros(3), "At least two"),
    ],
)
def test_analyze_rejects_invalid_shapes(two_factor_problem, X, Y, message):
    with pytest.raises(ValueError, match=message):
        shapley.analyze(two_factor_problem, X, Y)


def test_analyze_rejects_non_finite_values(two_factor_problem):
    X = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 2.0],
            [3.0, np.nan],
        ]
    )

    with pytest.raises(ValueError, match="finite"):
        shapley.analyze(two_factor_problem, X, np.zeros(6))
