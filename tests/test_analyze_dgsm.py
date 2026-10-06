from SALib.analyze import dgsm
from SALib.sample import finite_diff
from SALib.test_functions import Ishigami
from numpy.testing import assert_allclose
import numpy as np


PROBLEM = {
    "num_vars": 3,
    "names": ["x1", "x2", "x3"],
    "bounds": [[-np.pi, np.pi]] * 3,
}


def test_dgsm_conf_does_not_depend_on_output_scale():
    # dgsm is unitless, so its confidence interval must not change when Y is scaled
    X = finite_diff.sample(PROBLEM, 1000, seed=1)
    Y = Ishigami.evaluate(X)

    Si = dgsm.analyze(PROBLEM, X, Y, seed=1)
    Si_scaled = dgsm.analyze(PROBLEM, X, 10 * Y, seed=1)

    assert_allclose(Si_scaled["dgsm"], Si["dgsm"])
    assert_allclose(Si_scaled["dgsm_conf"], Si["dgsm_conf"])
