import numpy as np
import pytest

from SALib.analyze import pawn
from SALib.util import read_param_file
from SALib import ProblemSpec
from SALib.test_functions import Ishigami


def test_analyze_pawn():
    param_file = "src/SALib/test_functions/params/Ishigami_groups.txt"
    problem = read_param_file(param_file)
    sp = ProblemSpec(problem)
    sp.sample_sobol(512).evaluate(Ishigami.evaluate).analyze_pawn()


@pytest.mark.parametrize("kwargs, expected", [({"S": 1}, 0.0), ({}, 0.625)])
def test_analyze_pawn_includes_maximum(kwargs, expected):
    problem = {"num_vars": 1, "names": ["x"], "bounds": [[0, 1]]}
    X = np.array([0.0, 0.25, 0.75, 1.0])[:, None]

    Si = pawn.analyze(problem, X, X[:, 0], **kwargs)

    np.testing.assert_allclose(Si["median"], [expected])
