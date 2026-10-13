import numpy as np
import pytest

from SALib.analyze import (
    delta,
    dgsm,
    discrepancy,
    enhanced_hdmr,
    fast,
    ff,
    hdmr,
    morris,
    pawn,
    radial_ee,
    rbd_fast,
    rsa,
    shapley,
    sobol,
    sobol_jansen,
)
from SALib.sample import fast_sampler, finite_diff, latin
from SALib.sample import morris as morris_sample
from SALib.sample import sobol as sobol_sample
from SALib.test_functions import Ishigami

PROBLEM = {
    "num_vars": 3,
    "names": ["x1", "x2", "x3"],
    "bounds": [[-np.pi, np.pi]] * 3,
}


def _given_data(method):
    def run(X, Y):
        return method.analyze(PROBLEM, X, Y, seed=1)

    return latin.sample(PROBLEM, 500, seed=1), run


CASES = {
    "sobol": (
        sobol_sample.sample(PROBLEM, 64, seed=1),
        lambda X, Y: sobol.analyze(PROBLEM, Y, seed=1),
    ),
    "fast": (
        fast_sampler.sample(PROBLEM, 200, seed=1),
        lambda X, Y: fast.analyze(PROBLEM, Y, seed=1),
    ),
    "morris": (
        morris_sample.sample(PROBLEM, 20, seed=1),
        lambda X, Y: morris.analyze(PROBLEM, X, Y, seed=1),
    ),
    "dgsm": (
        finite_diff.sample(PROBLEM, 100, seed=1),
        lambda X, Y: dgsm.analyze(PROBLEM, X, Y, seed=1),
    ),
    "rbd_fast": _given_data(rbd_fast),
    "pawn": _given_data(pawn),
    "rsa": (
        latin.sample(PROBLEM, 500, seed=1),
        lambda X, Y: rsa.analyze(PROBLEM, X, Y),
    ),
    "discrepancy": (
        latin.sample(PROBLEM, 500, seed=1),
        lambda X, Y: discrepancy.analyze(PROBLEM, X, Y),
    ),
}


@pytest.mark.parametrize("name", CASES)
def test_analyze_accepts_lists(name):
    """X and Y given as plain lists give the same result as NumPy arrays."""
    X, run = CASES[name]
    Y = Ishigami.evaluate(X)
    expected = run(X, Y)
    got = run(X.tolist(), Y.tolist())
    for key, value in expected.items():
        if isinstance(value, np.ndarray) and value.dtype.kind == "f":
            np.testing.assert_allclose(got[key], value, err_msg=key)


Y_ONLY = [fast, sobol, sobol_jansen]
X_AND_Y = [
    delta,
    dgsm,
    discrepancy,
    enhanced_hdmr,
    ff,
    hdmr,
    morris,
    pawn,
    radial_ee,
    rbd_fast,
    rsa,
    shapley,
]
# Required arguments besides problem, X and Y.
EXTRA_ARGS = {radial_ee: {"sample_sets": 1}, sobol_jansen: {"sample_sets": 1}}
BAD_INPUTS = {
    "dict": {"x1": [1.0, 2.0]},
    "set": {1.0, 2.0},
    "str": "1.0, 2.0",
    "None": None,
    "list of str": ["a", "b"],
}


@pytest.mark.parametrize("bad", BAD_INPUTS.values(), ids=BAD_INPUTS)
@pytest.mark.parametrize(
    "method, arg",
    [(m, "Y") for m in Y_ONLY] + [(m, a) for m in X_AND_Y for a in "XY"],
    ids=lambda p: p if isinstance(p, str) else p.__name__.rsplit(".", 1)[-1],
)
def test_analyze_rejects_non_numeric_input(method, arg, bad):
    """A dict, string or other non-numeric X or Y raises a TypeError naming it."""
    X = latin.sample(PROBLEM, 10, seed=1)
    Y = Ishigami.evaluate(X)
    args = {"X": X, "Y": Y}
    args[arg] = bad
    names = ["Y"] if method in Y_ONLY else ["X", "Y"]
    msg = f"{arg} must be array_like of numbers .* got {type(bad).__name__}"
    with pytest.raises(TypeError, match=msg):
        method.analyze(PROBLEM, *(args[n] for n in names), **EXTRA_ARGS.get(method, {}))
