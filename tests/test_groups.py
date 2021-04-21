"""High-level tests to ensure grouped analyses convert to Pandas DataFrame."""

from typing import Dict, List

import numpy as np
from numpy.testing import assert_equal, assert_allclose

from SALib import ProblemSpec


def example_func(x):
    """Example linear test function."""
    return np.sum(x, axis=1)


def test_sobol_group_analysis():
    """Ensure valid groupings are returned from the Sobol analysis method."""

    group_spec = ProblemSpec(
        {
            "names": ["P1", "P2", "P3", "P4", "P5", "P6"],
            "bounds": [
                [0.0, 1.0],
                [1.0, 0.75],
                [0.0, 0.2],
                [0.0, 0.2],
                [-1.0, 1.0],
                [1.0, 0.25],
            ],
            "dists": ["unif", "triang", "norm", "lognorm", "unif", "triang"],
            "groups": ["A", "B"] * 3,
        }
    )

    # fmt: off
    (group_spec  
        .sample_saltelli(128)
        .evaluate(example_func)
        .analyze_sobol()
    )
    # fmt: on

    ST, S1, S2 = group_spec.to_df()

    assert len(ST.index) == 2, "Unexpected number of groups"
    assert len(ST[ST.index == "A"]) == 1, "Could not find Group A"
    assert len(ST[ST.index == "B"]) == 1, "Could not find Group B"


def test_morris_group_analysis():
    """Ensure valid groupings are returned from the Morris analysis method.

    Note: $\mu$ and $\sigma$ values will be NaN. See [1].


    References
    ----------

    .. [1] Campolongo, F., Cariboni, J., Saltelli, A., 2007. 
           An effective screening design for sensitivity analysis of large models. 
           Environmental Modelling & Software, 22, 1509–1518.
           https://dx.doi.org/10.1016/j.envsoft.2006.10.004
    """
    group_spec = ProblemSpec(
        {
            "names": ["P1", "P2", "P3", "P4", "P5", "P6"],
            "bounds": [[-100.0, 100.0]] * 6,
            "groups": ["A", "B", "C"] * 2,
        }
    )

    # fmt: off
    (group_spec  
        .sample_morris(100)
        .evaluate(example_func)
        .analyze_morris()
    )
    # fmt: on

    res = group_spec.to_df()

    assert len(res.index) == 3, "Unexpected number of groups"
    assert len(res[res.index == "A"]) == 1, "Could not find Group A"
    assert len(res[res.index == "B"]) == 1, "Could not find Group B"
    assert len(res[res.index == "C"]) == 1, "Could not find Group C"


if __name__ == "__main__":
    test_sobol_group_analysis()
    test_morris_group_analysis()

