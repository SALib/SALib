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


def setup_G4():
    problem = {
        'num_vars': 20,
        'names': ['x{}'.format(i) for i in range(1, 21)],
        'bounds': [[0, 1] * 20]
    }

    a = np.array([100, 0, 100, 100, 100,
                  100, 1, 0, 100, 100,
                  0, 100, 100, 100, 1,
                  100, 100, 0, 100, 1])

    alpha = np.array([1, 4, 1, 1, 1,
                      1, 0.5, 3, 1, 1,
                      2, 1, 1, 1, 0.5,
                      1, 1, 1.5, 1, 0.5])

    analytic_vals = Sobol_G._calc_analytic(a, alpha, 20)

    return problem, alpha, a, analytic_vals


def setup_G10():
    problem = {
        'num_vars': 20,
        'names': ['x{}'.format(i) for i in range(1, 21)],
        'bounds': [[0, 1] * 20]
    }

    # a = np.array([100, 0, 100, 100, 100,
    #               100, 1, 10, 0, 0,
    #               9, 0, 100, 100, 4,
    #               100, 100, 7, 100, 2])

    # alpha = np.array([1, 4, 1, 1, 1,
    #                   1, 0.4, 3, 0.8, 0.7,
    #                   2, 1.3, 1, 1, 0.3,
    #                   1, 1, 1.5, 1, 0.6])

    a = np.array([100, 0, 100, 100, 100, 
              100, 0, 1, 0, 0, 
              1, 0, 100, 100, 0,
              100, 100, 3, 100, 0])

    alpha = np.array([1, 4, 1, 1, 1,
                    1, 0.4, 3, 0.8, 0.7,
                    2, 1.3, 1, 1, 0.5,
                    1, 1, 2.5, 1, 0.6])

    analytic_vals = Sobol_G._calc_analytic(a, alpha, 20)

    return problem, alpha, a, analytic_vals


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
    num_sample_sets = 16000
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


def test_sobol_oat():
    """Radial OAT with a sample size of 1 should be equivalent to OAT
    """
    num_sample_sets = 1
    problem, _, _, _ = setup_G4()

    r_X = r_sample(problem, num_sample_sets)

    assert len(r_X) == 21, "Number of parameter sets should equal `N*(p+1)`"

    # Also ensure same size array is returned when skipping rows.
    r_X = r_sample(problem, num_sample_sets, skip_num=1)
    assert len(r_X) == 21, "Number of parameter sets should equal `N*(p+1)`"


def test_mc_oat_sample():
    """Radial OAT with a sample size of 1 should be equivalent to OAT
    """
    num_sample_sets = 1
    problem, _, _, _ = setup_G4()

    r_X = rmc_sample(problem, num_sample_sets, seed=101)

    assert len(r_X) == 21, "Number of parameter sets should equal `N*(p+1)`"

    num_sample_sets = 5
    problem, _, _, _ = setup_G4()

    r_X = rmc_sample(problem, num_sample_sets, seed=101)

    assert len(r_X) == (num_sample_sets*21), "Number of parameter sets should equal `N*(p+1)`"


def test_mc_oat():
    """Radial OAT with a sample size of 1 should be equivalent to OAT.

    Generated sample size should be `p+1`, where `p` is number of parameters.

    References
    ----------
    .. [1] Herman, J.D., Kollat, J.B., Reed, P.M., Wagener, T., 2013. 
           Technical Note: Method of Morris effectively reduces the computational 
           demands of global sensitivity analysis for distributed watershed models. 
           Hydrology and Earth System Sciences 17, 2893–2903. 
           https://doi.org/10.5194/hess-17-2893-2013

    """
    num_sample_sets = 5000
    problem, alpha, a, analytic_vals = setup_G4()

    r_X = rmc_sample(problem, num_sample_sets, seed=42)

    delta = np.random.rand(20)
    r_Y = Sobol_G.evaluate(r_X, a, alpha=alpha, delta=delta)
    res = ree_analyze(problem, r_X, r_Y, num_sample_sets, seed=42)

    # Mu* does not give a quantitative interpretation of percent variance (Herman et al. 2013)
    # See Ref [1]
    # Here, we normalize to values between 0 - 1
    scaled = (res['mu_star'] / (res['mu_star'].max() - res['mu_star'].min())).round(2)

    assert np.array_equal(np.where(analytic_vals > 0.05), np.where(scaled > 0.15)), \
        "Mismatch in identified vs expected sensitive parameters."




if __name__ == '__main__':
    test_sobol_oat()
    test_mc_oat()

    test_radial_jansen_G4()
    test_radial_jansen_G10()


    
