import pytest
import copy

import numpy as np
import pandas as pd

from SALib import ProblemSpec
from SALib.test_functions import Ishigami


def test_sp_bounds():
    """Ensure incorrect user-defined bounds raises AssertionErrors."""
    with pytest.raises(AssertionError):
        ProblemSpec()

    with pytest.raises(AssertionError):
        ProblemSpec({"names": ["x1", "x2", "x3"], "groups": None, "outputs": ["Y"]})

    with pytest.raises(AssertionError):
        ProblemSpec(
            {
                "names": ["x1", "x2", "x3"],
                "groups": None,
                "bounds": [[-np.pi, np.pi] * 3],
                "outputs": ["Y"],
            }
        )


def test_sp():
    """Ensure basic usage works without raising any errors."""

    sp = ProblemSpec(
        {
            "names": ["x1", "x2", "x3"],
            "groups": None,
            "bounds": [[-np.pi, np.pi]] * 3,
            "outputs": ["Y"],
        }
    )

    (
        sp.sample_sobol(128, calc_second_order=True)
        .evaluate(Ishigami.evaluate)
        .analyze_sobol(calc_second_order=True, conf_level=0.95)
    )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_sp_setters():
    """Test sample and result setters."""

    sp = ProblemSpec(
        {
            "names": ["x1", "x2", "x3"],
            "groups": None,
            "bounds": [[-np.pi, np.pi]] * 3,
            "outputs": ["Y"],
        }
    )

    nvars = sp["num_vars"]
    N = 128
    X1 = sp.sample_sobol(N, calc_second_order=True).samples

    # Saltelli produces `N*(2p+2)` samples when producing samples for 2nd order analysis
    expected_samples = 128 * (2 * nvars + 2)

    assert X1.shape[0] == expected_samples, "Number of samples is not as expected"

    # Get another 100 samples
    X2 = sp.sample_sobol(128, calc_second_order=True).samples
    assert np.all(
        sp.samples == X2
    ), "Stored sample and extracted samples are not identical!"

    # Test property setter
    # this approach actually overrides the underlying sp._samples attribute
    sp.samples = X1
    assert np.all(sp.samples == X1), "Reverting samples to original set did not work!"

    # Setting samples should clear results attribute
    sp.set_samples(X2)
    Y2 = sp.evaluate(Ishigami.evaluate).results
    sp.set_samples(X1)
    assert (
        sp.results is None
    ), "Results were not cleared after new sample values were set!"

    # Entire process should not raise errors
    (
        sp.sample_saltelli(128, calc_second_order=True)
        .set_samples(X2)
        .set_results(Y2)
        .analyze_sobol(calc_second_order=True, conf_level=0.95)
    )


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_parallel_single_output():
    # Create the SALib Problem specification
    sp = ProblemSpec(
        {
            "names": ["x1", "x2", "x3"],
            "groups": None,
            "bounds": [[-np.pi, np.pi]] * 3,
            "outputs": ["Y"],
        }
    )

    # Single core example
    (
        sp.sample_saltelli(2**8)
        .evaluate(Ishigami.evaluate)
        .analyze_sobol(calc_second_order=True, conf_level=0.95, seed=101)
    )

    # Parallel example
    psp = copy.deepcopy(sp)
    (
        psp.sample_saltelli(2**8)
        .evaluate_parallel(Ishigami.evaluate, nprocs=2)
        .analyze_sobol(calc_second_order=True, conf_level=0.95, nprocs=2, seed=101)
    )

    assert (
        np.testing.assert_allclose(sp.results, psp.results, rtol=1e-3, atol=1e-3)
        is None
    ), "Model results not equal!"

    for col, x in sp.analysis.items():
        assert (
            np.testing.assert_allclose(x, psp.analysis[col]) is None
        ), "Analysis results not equal!"


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_parallel_multi_output():
    from SALib.test_functions import lake_problem

    # Create the SALib Problem specification
    sp = ProblemSpec(
        {
            "names": ["a", "q", "b", "mean", "stdev", "delta", "alpha"],
            "bounds": [
                [0.0, 0.1],
                [2.0, 4.5],
                [0.1, 0.45],
                [0.01, 0.05],
                [0.001, 0.005],
                [0.93, 0.99],
                [0.2, 0.5],
            ],
            "outputs": ["max_P", "Utility", "Inertia", "Reliability"],
        }
    )

    # Single core example
    (
        sp.sample_saltelli(2**8)
        .evaluate(lake_problem.evaluate)
        .analyze_sobol(calc_second_order=True, conf_level=0.95, seed=101)
    )

    # Parallel example
    psp = copy.deepcopy(sp)
    (
        psp.sample_saltelli(2**8)
        .evaluate_parallel(lake_problem.evaluate, nprocs=2)
        .analyze_sobol(calc_second_order=True, conf_level=0.95, nprocs=2, seed=101)
    )

    x_df = pd.DataFrame(sp.results)
    y_df = pd.DataFrame(psp.results)
    for col in pd.DataFrame(sp.results):
        z = [
            np.testing.assert_allclose(x, y)
            for x, y in zip(x_df.loc[:, col], y_df.loc[:, col])
        ]

        assert not any(z), "Model results are not equal!"

    x_df = pd.DataFrame(sp.analysis)
    y_df = pd.DataFrame(psp.analysis)
    for col in pd.DataFrame(sp.analysis):
        z = [
            np.testing.assert_allclose(x, y)
            for x, y in zip(x_df.loc[:, col], y_df.loc[:, col])
        ]

        assert not any(z), "Analysis results are not equal!"


def test_single_parameter():
    """Ensures error is raised when attempting to conduct SA on a single parameter."""

    sp = ProblemSpec({"names": ["x1"], "bounds": [[-1, 1]], "outputs": ["Y"]})

    def func(X):
        return X[1] * X[0]

    sp.sample_latin(50)
    sp.evaluate(func)

    with pytest.raises(ValueError):
        sp.analyze_hdmr()


def test_single_group():
    """Ensures error is raised when attempting to conduct SA on a single group."""
    sp = ProblemSpec(
        {"names": ["x1"], "bounds": [[-1, 1]], "groups": ["G1"], "outputs": ["Y"]}
    )

    def func(X):
        return X[1] * X[0]

    sp.sample_latin(50)
    sp.evaluate(func)

    with pytest.raises(ValueError):
        sp.analyze_hdmr()
