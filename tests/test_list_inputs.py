import numpy as np
import pytest

from SALib import ProblemSpec
from SALib.test_functions import Ishigami


def _spec():
    return ProblemSpec({"names": ["x1", "x2", "x3"], "bounds": [[-np.pi, np.pi]] * 3})


@pytest.mark.parametrize(
    "sampler, analyzer, keys",
    [
        ("sample_sobol", "analyze_sobol", ["S1", "ST"]),
        ("sample_latin", "analyze_delta", ["delta_raw", "S1"]),
    ],
)
def test_problemspec_accepts_lists(sampler, analyzer, keys):
    """Samples and results given as lists give the same indices as arrays."""
    X = getattr(_spec(), sampler)(64, seed=1).samples
    Y = Ishigami.evaluate(X)

    def analyze(X, Y):
        sp = _spec().set_samples(X).set_results(Y)
        return getattr(sp, analyzer)(seed=1).analysis

    expected = analyze(X, Y)
    got = analyze(X.tolist(), Y.tolist())
    for key in keys:
        np.testing.assert_allclose(got[key], expected[key], err_msg=key)


def test_evaluate_accepts_list_output():
    """A model that returns a list is stored as an array."""
    sp = _spec().sample_latin(50, seed=1)
    sp.evaluate(lambda X: Ishigami.evaluate(X).tolist())
    np.testing.assert_array_equal(sp.results, Ishigami.evaluate(sp.samples))


BAD_INPUTS = {
    "dict": {"x1": [1.0, 2.0]},
    "set": {1.0, 2.0},
    "str": "1.0, 2.0",
    "None": None,
    "list of str": ["a", "b"],
}


@pytest.mark.parametrize("bad", BAD_INPUTS.values(), ids=BAD_INPUTS)
@pytest.mark.parametrize("setter", ["set_samples", "set_results"])
def test_problemspec_rejects_non_numeric_input(setter, bad):
    """A dict, string or other non-numeric input raises a TypeError naming it."""
    name = setter.removeprefix("set_")
    msg = f"{name} must be array_like of numbers .* got {type(bad).__name__}"
    with pytest.raises(TypeError, match=msg):
        getattr(_spec(), setter)(bad)
