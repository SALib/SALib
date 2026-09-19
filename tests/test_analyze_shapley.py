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


def test_analyze_rejects_groups(two_factor_problem):
    two_factor_problem["groups"] = ["A", "B"]
    X = np.zeros((6, 2))
    Y = np.zeros(6)

    with pytest.raises(ValueError, match="does not support groups"):
        shapley.analyze(two_factor_problem, X, Y)


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
