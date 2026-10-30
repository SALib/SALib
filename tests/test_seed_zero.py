import numpy as np
import pytest

from SALib.analyze import delta, dgsm, enhanced_hdmr, fast, hdmr, radial_ee, sobol
from SALib.analyze import sobol_jansen
from SALib.sample import fast_sampler, finite_diff, latin
from SALib.sample import sobol as sobol_sample
from SALib.sample.radial import radial_mc
from SALib.test_functions import Ishigami

PROBLEM = {
    "num_vars": 3,
    "names": ["x1", "x2", "x3"],
    "bounds": [[-np.pi, np.pi]] * 3,
}


def _sobol(seed):
    X = sobol_sample.sample(PROBLEM, 64, seed=1)
    return sobol.analyze(PROBLEM, Ishigami.evaluate(X), seed=seed)["S1_conf"]


def _fast(seed):
    X = fast_sampler.sample(PROBLEM, 200, seed=1)
    return fast.analyze(PROBLEM, Ishigami.evaluate(X), seed=seed)["S1_conf"]


def _delta(seed):
    X = latin.sample(PROBLEM, 500, seed=1)
    Y = Ishigami.evaluate(X)
    return delta.analyze(PROBLEM, X, Y, num_resamples=10, seed=seed)["delta_raw"]


def _dgsm(seed):
    X = finite_diff.sample(PROBLEM, 100, seed=1)
    return dgsm.analyze(PROBLEM, X, Ishigami.evaluate(X), seed=seed)["dgsm_conf"]


def _hdmr(seed):
    X = latin.sample(PROBLEM, 300, seed=1)
    Y = Ishigami.evaluate(X)
    with pytest.warns(DeprecationWarning):
        Si = hdmr.analyze(PROBLEM, X, Y, maxorder=1, K=5, seed=seed)
    return Si["ST_conf"]


def _enhanced_hdmr(seed):
    X = latin.sample(PROBLEM, 300, seed=1)
    Y = Ishigami.evaluate(X)
    # A copy of the names: with max_order > 1, enhanced_hdmr appends the
    # interaction terms to the list it is given.
    problem = {**PROBLEM, "names": list(PROBLEM["names"])}
    Si = enhanced_hdmr.analyze(problem, X, Y, max_order=2, bootstrap=5, seed=seed)
    return Si["S"]


def _radial_mc(seed):
    return radial_mc.sample(PROBLEM, 10, seed=seed)


# Sampled once here: radial_mc.sample reseeds NumPy's global generator, which
# would hide an ignored seed in the analyze call that follows it.
RADIAL_X = radial_mc.sample(PROBLEM, 10, seed=1)
RADIAL_Y = Ishigami.evaluate(RADIAL_X)


def _radial_ee(seed):
    Si = radial_ee.analyze(PROBLEM, RADIAL_X, RADIAL_Y, 10, seed=seed)
    return Si["mu_star_conf"]


def _sobol_jansen(seed):
    Si = sobol_jansen.analyze(PROBLEM, RADIAL_Y, 10, seed=seed)
    return Si["ST_conf"]


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize(
    "run",
    [
        _sobol,
        _fast,
        _delta,
        _dgsm,
        _hdmr,
        _enhanced_hdmr,
        _radial_mc,
        _radial_ee,
        _sobol_jansen,
    ],
)
def test_seed_is_reproducible(run, seed):
    """Every seed, 0 included, must give the same result twice."""
    np.testing.assert_array_equal(run(seed), run(seed))
