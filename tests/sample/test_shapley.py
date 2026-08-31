import numpy as np
import pytest

from SALib.sample import shapley


@pytest.fixture
def problem():
    return {
        "num_vars": 3,
        "names": ["x1", "x2", "x3"],
        "bounds": [[-1.0, 1.0], [0.0, 2.0], [10.0, 20.0]],
    }


def test_sample_shape_and_trajectory_structure(problem):
    samples = shapley.sample(problem, 8, seed=42)

    assert samples.shape == (32, 3)
    trajectories = samples.reshape(8, 4, 3)
    for trajectory in trajectories:
        changed = [
            np.flatnonzero(trajectory[step + 1] != trajectory[step])
            for step in range(3)
        ]
        assert all(factors.size == 1 for factors in changed)
        assert sorted(int(factors[0]) for factors in changed) == [0, 1, 2]


def test_sample_seed_is_reproducible(problem):
    first = shapley.sample(problem.copy(), 4, seed=101)
    second = shapley.sample(problem.copy(), 4, seed=101)

    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("N", [0, -1, 1.5, True])
def test_sample_rejects_invalid_sample_count(problem, N):
    with pytest.raises(ValueError, match="positive integer"):
        shapley.sample(problem, N, seed=1)


def test_sample_rejects_groups(problem):
    problem["groups"] = ["A", "A", "B"]

    with pytest.raises(ValueError, match="does not support groups"):
        shapley.sample(problem, 4, seed=1)
