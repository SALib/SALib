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

    assert_allclose(result["Shapley"], [69.0, 37.0])
    assert_allclose(
        result["Shapley_conf"],
        norm.ppf(0.975) * np.array([61.0, 5.0]),
    )
    assert result["names"] == ["x1", "x2"]


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
    assert_allclose(result["Shapley"], expected, atol=0.18)

    trajectory_outputs = Y.reshape(20_000, 4)
    variance_estimate = np.mean(
        (trajectory_outputs[:, 0] - trajectory_outputs[:, -1]) ** 2 / 2
    )
    assert_allclose(np.sum(result["Shapley"]), variance_estimate)


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
