"""The bootstrap in sobol.analyze must match resampling by fancy indexing."""

import numpy as np
from numpy.testing import assert_allclose
from pytest import mark, warns
from scipy.stats import norm

from SALib.analyze import sobol
from SALib.analyze.sobol import (
    first_order,
    total_order,
    second_order,
    separate_output_values,
)
from SALib.sample import sobol as sobol_sampler
from SALib.test_functions import Ishigami, Sobol_G
from SALib.util import extract_group_names

ISHIGAMI = {
    "num_vars": 3,
    "names": ["x1", "x2", "x3"],
    "bounds": [[-np.pi, np.pi]] * 3,
}
GROUPED = dict(ISHIGAMI, groups=["G1", "G2", "G1"])
SOBOL_G = {
    "num_vars": 6,
    "names": [f"x{i}" for i in range(1, 7)],
    "bounds": [[0, 1]] * 6,
}
G_A = np.array([78, 12, 0.5, 2, 97, 33])


def reference_analyze(problem, Y, calc_second_order, num_resamples, conf_level, seed):
    """sobol.analyze as it was, resampling with A[r], AB[r, j] and B[r]."""
    rng = (
        np.random.default_rng(seed).integers if seed is not None else np.random.randint
    )
    _, D = extract_group_names(problem)
    N = Y.size // (2 * D + 2 if calc_second_order else D + 2)
    Y = (Y - Y.mean()) / Y.std()
    A, B, AB, BA = separate_output_values(Y, D, N, calc_second_order)
    r = rng(N, size=(N, num_resamples))
    Z = norm.ppf(0.5 + conf_level / 2)
    S = {k: np.zeros(D) for k in ("S1", "S1_conf", "ST", "ST_conf")}
    S["S1_conf_all"] = np.zeros((num_resamples, D))
    S["ST_conf_all"] = np.zeros((num_resamples, D))
    for j in range(D):
        S["S1"][j] = first_order(A, AB[:, j], B)
        S["S1_conf_all"][:, j] = first_order(A[r], AB[r, j], B[r])
        S["ST"][j] = total_order(A, AB[:, j], B)
        S["ST_conf_all"][:, j] = total_order(A[r], AB[r, j], B[r])
        if np.ptp(np.r_[A[r], B[r]]) != 0.0:
            S["S1_conf"][j] = Z * S["S1_conf_all"][:, j].std(ddof=1)
            S["ST_conf"][j] = Z * S["ST_conf_all"][:, j].std(ddof=1)
    if calc_second_order:
        S["S2"] = np.full((D, D), np.nan)
        S["S2_conf"] = np.full((D, D), np.nan)
        for j in range(D):
            for k in range(j + 1, D):
                S["S2"][j, k] = second_order(A, AB[:, j], AB[:, k], BA[:, j], B)
                S["S2_conf"][j, k] = Z * second_order(
                    A[r], AB[r, j], AB[r, k], BA[r, j], B[r]
                ).std(ddof=1)
    return S


def model_output(name, calc_second_order, N):
    problem = {"ishigami": ISHIGAMI, "grouped": GROUPED, "sobol_g": SOBOL_G}[name]
    X = sobol_sampler.sample(problem, N, calc_second_order=calc_second_order)
    if name == "sobol_g":
        return problem, Sobol_G.evaluate(X, G_A)
    return problem, Ishigami.evaluate(X)


def assert_matches(problem, Y, c2o, seed, num_resamples=100, conf_level=0.95, **kw):
    if seed is None:
        np.random.seed(3)
    expected = reference_analyze(problem, Y, c2o, num_resamples, conf_level, seed)
    if seed is None:
        np.random.seed(3)
    actual = sobol.analyze(
        problem,
        Y,
        calc_second_order=c2o,
        num_resamples=num_resamples,
        conf_level=conf_level,
        seed=seed,
        **kw,
    )
    assert ("S1_conf_all" in actual) == kw.get("keep_resamples", False)
    for key in actual:
        assert_allclose(actual[key], expected[key], atol=1e-12, rtol=0, err_msg=key)


@mark.filterwarnings("ignore:Second-order indices")
@mark.parametrize("name", ["ishigami", "grouped", "sobol_g"])
@mark.parametrize("c2o", [True, False])
@mark.parametrize("seed", [7, None])
def test_bootstrap_matches_reference(name, c2o, seed):
    problem, Y = model_output(name, c2o, 256)
    assert_matches(problem, Y, c2o, seed, keep_resamples=True)


def test_bootstrap_conf_level_and_resamples():
    problem, Y = model_output("ishigami", True, 64)
    assert_matches(
        problem, Y, True, 11, num_resamples=37, conf_level=0.8, keep_resamples=True
    )


@mark.parametrize("c2o", [True, False])
def test_bootstrap_parallel_matches_reference(c2o):
    problem, Y = model_output("sobol_g", c2o, 128)
    assert_matches(problem, Y, c2o, 5, parallel=True)


@mark.filterwarnings("ignore::UserWarning")
@mark.parametrize("c2o", [True, False])
@mark.parametrize("which", ["all", "A_and_B"])
def test_bootstrap_constant_output(c2o, which):
    problem, Y = model_output("ishigami", c2o, 64)
    if which == "all":
        Y = np.full_like(Y, 0.4)
    else:
        step = 8 if c2o else 5
        Y[0::step] = 0.4
        Y[step - 1 :: step] = 0.4
    assert_matches(problem, Y, c2o, 2, keep_resamples=True)
    assert_matches(problem, Y, c2o, 2, parallel=True)
    if which == "A_and_B":
        with warns(UserWarning, match="Constant values"):
            sobol.analyze(problem, Y, calc_second_order=c2o, seed=2)


def test_bootstrap_counts_skips_undrawn_rows():
    A = np.array([0.0, 1.0, 2.0, 30.0])
    B = A + 1
    r = np.array([[0, 1], [1, 0], [2, 2], [0, 1]])  # row 3 is never drawn
    _, y_var, y_ptp = sobol._bootstrap_counts(A, B, r)
    assert y_ptp == np.ptp(np.r_[A[r], B[r]])
    assert_allclose(y_var, np.var(np.r_[A[r], B[r]], axis=0), atol=1e-12, rtol=0)
