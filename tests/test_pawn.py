import sys
sys.path.append('../..')

from SALib.sample import latin
from SALib.analyze import pawn
from SALib.test_functions import Ishigami

import numpy as np


def test_pawn_analyze():
    """Confirms PAWN indices are in line with reported expected values.

    Expected values are taken from [1] for the Ishigami test function.


    References
    ----------
    .. [1] Pianosi, F., Wagener, T., 2018. 
           Distribution-based sensitivity analysis from a generic input-output 
           sample. 
           Environmental Modelling & Software 108, 197–207. 
           https://doi.org/10.1016/j.envsoft.2018.07.019
    """

    problem = {
               'num_vars': 3,
               'names': ['x1', 'x2', 'x3'],
               'bounds': [[-np.pi, np.pi]*3]
    }
    X = latin.sample(problem, 500)
    Y = Ishigami.evaluate(X, A=2.0, B=1.0)
    Si = pawn.analyze(problem, X, Y, S=10, stat='median', 
                      num_resamples=100)

    Si_df = Si.to_df()

    bnd = Si_df['PAWNi_conf'].values.T
    p_i = Si_df['PAWNi']

    expected_vals = np.array([0.5, 0.16, 0.29])

    in_bounds = ((p_i >= (expected_vals - bnd)) & (p_i <= (expected_vals + bnd)))

    assert np.all(in_bounds)
