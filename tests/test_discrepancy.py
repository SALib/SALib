import numpy as np
from numpy.testing import assert_allclose

from SALib.sample import latin
from SALib.analyze import discrepancy
from SALib.test_functions import Ishigami


def test_discrepancy():
    problem = {
        "num_vars": 3,
        "names": ["x1", "x2", "x3"],
        "bounds": [[-np.pi, np.pi]] * 3,
    }
    X = latin.sample(problem, 1_000)
    Y = Ishigami.evaluate(X)
    Si = discrepancy.analyze(problem, X, Y)

    assert_allclose(Si["s_discrepancy"], [0.33, 0.33, 0.33], atol=5e-3)
