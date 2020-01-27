"""Tests for Radial OAT using Sobol sampling"""

from pytest import raises
from numpy.testing import assert_equal, assert_allclose
import numpy as np

from SALib.util import scale_samples
from SALib.sample import sobol_sequence, common_args
from SALib.sample.radial.radial_sobol import sample as r_sample
from SALib.sample.radial.radial_mc import sample as rmc_sample
from SALib.analyze.sobol_jansen import analyze as j_analyze
from SALib.analyze.radial_ee import analyze as ree_analyze

from SALib.test_functions import Sobol_G
from test_setup import setup_G10, setup_G4


def test_radial_jansen_G4():
    """Calculate and compare analytic values against reported ST values for G*4.

    We use the values for `a` and `alpha` from [1], but note that the analytic values
    provided in the paper are incorrect.

    References
    ----------
    .. [1] Campolongo, F., Saltelli, A., Cariboni, J., 2011. 
           From screening to quantitative sensitivity analysis: A unified approach. 
           Computer Physics Communications 182, 978–988.
           https://www.sciencedirect.com/science/article/pii/S0010465510005321
           DOI: 10.1016/j.cpc.2010.12.039
    """
    num_sample_sets = 15000
    problem, alpha, a, analytic_vals = setup_G4()

    delta = np.random.rand(20)
    r_X = r_sample(problem, num_sample_sets)
    r_Y = Sobol_G.evaluate(r_X, a, alpha=alpha, delta=delta)

    g4_result = j_analyze(problem, r_Y, num_sample_sets,
                          num_resamples=1000)
    ST_res = g4_result['ST'].round(4)

    assert_allclose(analytic_vals, ST_res, atol=0.04, rtol=0.04)

    num_analytic = len(np.where(analytic_vals > 0.05)[0])
    assert num_analytic == 4, \
        "Expected 4 analytic values, but only found {}".format(num_analytic)

    assert np.array_equal(np.where(analytic_vals > 0.05), np.where(ST_res > 0.05)), \
        "Mismatch in identified vs expected sensitive parameters."


def test_radial_jansen_G10():
    """Calculate and compare analytic values against reported ST values for G*10.

    We use the values for `a` and `alpha` from [1], but note that the analytic values
    provided in the paper are incorrect.

    References
    ----------
    .. [1] Campolongo, F., Saltelli, A., Cariboni, J., 2011. 
           From screening to quantitative sensitivity analysis: A unified approach. 
           Computer Physics Communications 182, 978–988.
           https://www.sciencedirect.com/science/article/pii/S0010465510005321
           DOI: 10.1016/j.cpc.2010.12.039
    """
    num_sample_sets = 20000
    problem, alpha, a, analytic_vals = setup_G10()

    delta = np.random.rand(20)
    r_X = r_sample(problem, num_sample_sets)
    r_Y = Sobol_G.evaluate(r_X, a, alpha=alpha, delta=delta)

    g10_result = j_analyze(problem, r_Y, num_sample_sets,
                           num_resamples=1000)
    ST_res = g10_result['ST'].round(4)

    assert_allclose(analytic_vals, ST_res, atol=0.04, rtol=0.04)

    num_analytic = len(np.where(analytic_vals > 0.05)[0])
    assert num_analytic == 10, \
        "Expected 10 analytic values, but only found {}".format(num_analytic)

    assert np.array_equal(np.where(analytic_vals > 0.05), np.where(ST_res > 0.05)), \
        "Mismatch in identified vs expected sensitive parameters."



if __name__ == '__main__':
    test_radial_jansen_G4()
    test_radial_jansen_G10()


    
