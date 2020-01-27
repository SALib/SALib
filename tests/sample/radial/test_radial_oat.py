import numpy as np

from SALib.sample.radial.radial_sobol import sample as r_sample
from SALib.sample.radial.radial_mc import sample as rmc_sample
from SALib.analyze.radial_ee import analyze as ree_analyze
from SALib.test_functions import Sobol_G

from test_setup import setup_G10, setup_G4


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

    test_mc_oat_sample()
    test_mc_oat()