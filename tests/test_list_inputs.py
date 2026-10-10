import numpy as np
import pytest

from SALib.analyze import dgsm, discrepancy, fast, morris, pawn, rbd_fast, rsa, sobol
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
