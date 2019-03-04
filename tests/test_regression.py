from __future__ import division

from numpy.testing import assert_allclose

from SALib.analyze import delta
from SALib.analyze import dgsm
from SALib.analyze import fast
from SALib.analyze import rbd_fast
from SALib.analyze import sobol
from SALib.sample import fast_sampler
from SALib.sample import finite_diff
from SALib.sample import latin
from SALib.sample import saltelli
import numpy as np

from SALib.analyze import morris
from SALib.sample.morris import sample
from SALib.test_functions import Ishigami
from SALib.util import read_param_file

from pytest import fixture


@fixture(scope='function')
def set_seed():
    """Sets seeds for random generators so that tests can be repeated

    It is necessary to set seeds for both the numpy.random, and
    the stdlib.random libraries.
    """
    seed = 123456
    np.random.seed(seed)


class TestMorris:

    def test_regression_morris_vanilla(self, set_seed):
        """Note that this is a poor estimate of the Ishigami
        function.
        """
        set_seed
        param_file = 'src/SALib/test_functions/params/Ishigami.txt'
        problem = read_param_file(param_file)
        param_values = sample(problem, 10000, 4,
                              optimal_trajectories=None)

        Y = Ishigami.evaluate(param_values)

        Si = morris.analyze(problem, param_values, Y,
                            conf_level=0.95, print_to_console=False,
                            num_levels=4)

        assert_allclose(Si['mu_star'], [7.536586, 7.875, 6.308785],
                        atol=0, rtol=1e-5)

    def test_regression_morris_groups(self, set_seed):
        set_seed
        param_file = 'src/SALib/test_functions/params/Ishigami_groups.txt'
        problem = read_param_file(param_file)

        param_values = sample(problem=problem, N=10000,
                              num_levels=4,
                              optimal_trajectories=None)

        Y = Ishigami.evaluate(param_values)

        Si = morris.analyze(problem, param_values, Y,
                            conf_level=0.95, print_to_console=False,
                            num_levels=4)

        assert_allclose(Si['mu_star'], [7.610322, 10.197014],
                        atol=0, rtol=1e-5)

    def test_regression_morris_groups_brute_optim(self, set_seed):

        set_seed
        param_file = 'src/SALib/test_functions/params/Ishigami_groups.txt'
        problem = read_param_file(param_file)

        param_values = sample(problem=problem, N=50,
                              num_levels=4,
                              optimal_trajectories=6,
                              local_optimization=False)

        Y = Ishigami.evaluate(param_values)

        Si = morris.analyze(problem, param_values, Y,
                            conf_level=0.95, print_to_console=False,
                            num_levels=4)

        assert_allclose(Si['mu'], [9.786986, np.NaN],
                        atol=0, rtol=1e-5)

        assert_allclose(Si['sigma'], [6.453729, np.NaN],
                        atol=0, rtol=1e-5)

        assert_allclose(Si['mu_star'], [9.786986, 7.875],
                        atol=0, rtol=1e-5)

    def test_regression_morris_groups_local_optim(self, set_seed):
        set_seed
        param_file = 'src/SALib/test_functions/params/Ishigami_groups.txt'
        problem = read_param_file(param_file)

        param_values = sample(problem=problem, N=500,
                              num_levels=4,
                              optimal_trajectories=20,
                              local_optimization=True)

        Y = Ishigami.evaluate(param_values)

        Si = morris.analyze(problem, param_values, Y,
                            conf_level=0.95, print_to_console=False,
                            num_levels=4)

        assert_allclose(Si['mu_star'],
                        [13.95285, 7.875],
                        rtol=1e-5)

    def test_regression_morris_optimal(self, set_seed):
        '''
        Tests the use of optimal trajectories with Morris.

        Uses brute force approach

        Note that the relative tolerance is set to a very high value
        (default is 1e-05) due to the coarse nature of the num_levels.
        '''
        set_seed
        param_file = 'src/SALib/test_functions/params/Ishigami.txt'
        problem = read_param_file(param_file)
        param_values = sample(problem=problem, N=20,
                              num_levels=4,
                              optimal_trajectories=9,
                              local_optimization=True)

        Y = Ishigami.evaluate(param_values)

        Si = morris.analyze(problem, param_values, Y,
                            conf_level=0.95, print_to_console=False,
                            num_levels=4)

        assert_allclose(Si['mu_star'],
                        [9.786986e+00, 7.875000e+00, 1.388621],
                        atol=0,
                        rtol=1e-5)


def test_regression_sobol():
    param_file = 'src/SALib/test_functions/params/Ishigami.txt'
    problem = read_param_file(param_file)
    param_values = saltelli.sample(problem, 10000, calc_second_order=True)

    Y = Ishigami.evaluate(param_values)

    Si = sobol.analyze(problem, Y,
                       calc_second_order=True, conf_level=0.95,
                       print_to_console=False)

    assert_allclose(Si['S1'], [0.31, 0.44, 0.00], atol=5e-2, rtol=1e-1)
    assert_allclose(Si['ST'], [0.55, 0.44, 0.24], atol=5e-2, rtol=1e-1)
    assert_allclose([Si['S2'][0][1], Si['S2'][0][2], Si['S2'][1][2]], [
                    0.00, 0.25, 0.00], atol=5e-2, rtol=1e-1)


def test_regression_sobol_parallel():
    param_file = 'src/SALib/test_functions/params/Ishigami.txt'
    problem = read_param_file(param_file)
    param_values = saltelli.sample(problem, 10000, calc_second_order=True)

    Y = Ishigami.evaluate(param_values)

    Si = sobol.analyze(problem, Y,
                       calc_second_order=True, parallel=True,
                       conf_level=0.95, print_to_console=False)

    assert_allclose(Si['S1'], [0.31, 0.44, 0.00], atol=5e-2, rtol=1e-1)
    assert_allclose(Si['ST'], [0.55, 0.44, 0.24], atol=5e-2, rtol=1e-1)
    assert_allclose([Si['S2'][0][1], Si['S2'][0][2], Si['S2'][1][2]], [
                    0.00, 0.25, 0.00], atol=5e-2, rtol=1e-1)


def test_regression_sobol_groups():
    problem = {
        'num_vars': 3,
        'names': ['x1', 'x2', 'x3'],
        'bounds': [[-np.pi, np.pi]] * 3,
        'groups': ['G1', 'G2', 'G1']
    }
    param_values = saltelli.sample(problem, 10000, calc_second_order=True)

    Y = Ishigami.evaluate(param_values)
    Si = sobol.analyze(problem, Y,
                       calc_second_order=True, parallel=True,
                       conf_level=0.95, print_to_console=False)

    assert_allclose(Si['S1'], [0.55, 0.44], atol=5e-2, rtol=1e-1)
    assert_allclose(Si['ST'], [0.55, 0.44], atol=5e-2, rtol=1e-1)
    assert_allclose(Si['S2'][0][1], [0.00], atol=5e-2, rtol=1e-1)


def test_regression_sobol_groups_dists():
    problem = {
        'num_vars': 3,
        'names': ['x1', 'x2', 'x3'],
        'bounds': [[-np.pi, np.pi], [1.0, 0.2], [3, 0.5]],
        'groups': ['G1', 'G2', 'G1'],
        'dists': ['unif', 'lognorm', 'triang']
    }
    param_values = saltelli.sample(problem, 10000, calc_second_order=True)

    Y = Ishigami.evaluate(param_values)
    Si = sobol.analyze(problem, Y,
                       calc_second_order=True, parallel=True,
                       conf_level=0.95, print_to_console=False)

    assert_allclose(Si['S1'], [0.427, 0.573], atol=5e-2, rtol=1e-1)
    assert_allclose(Si['ST'], [0.428, 0.573], atol=5e-2, rtol=1e-1)
    assert_allclose(Si['S2'][0][1], [0.001], atol=5e-2, rtol=1e-1)


def test_regression_fast():
    param_file = 'src/SALib/test_functions/params/Ishigami.txt'
    problem = read_param_file(param_file)
    param_values = fast_sampler.sample(problem, 10000)

    Y = Ishigami.evaluate(param_values)

    Si = fast.analyze(problem, Y, print_to_console=False)
    assert_allclose(Si['S1'], [0.31, 0.44, 0.00], atol=5e-2, rtol=1e-1)
    assert_allclose(Si['ST'], [0.55, 0.44, 0.24], atol=5e-2, rtol=1e-1)


def test_regression_rbd_fast():
    param_file = 'src/SALib/test_functions/params/Ishigami.txt'
    problem = read_param_file(param_file)
    param_values = latin.sample(problem, 10000)

    Y = Ishigami.evaluate(param_values)

    Si = rbd_fast.analyze(problem, Y, param_values, print_to_console=False)
    assert_allclose(Si['S1'], [0.31, 0.44, 0.00], atol=5e-2, rtol=1e-1)


def test_regression_dgsm():
    param_file = 'src/SALib/test_functions/params/Ishigami.txt'
    problem = read_param_file(param_file)
    param_values = finite_diff.sample(problem, 10000, delta=0.001)

    Y = Ishigami.evaluate(param_values)

    Si = dgsm.analyze(problem, param_values, Y,
                      conf_level=0.95, print_to_console=False)

    assert_allclose(Si['dgsm'], [2.229, 7.066, 3.180], atol=5e-2, rtol=1e-1)


def test_regression_delta():
    param_file = 'src/SALib/test_functions/params/Ishigami.txt'
    problem = read_param_file(param_file)
    param_values = latin.sample(problem, 10000)

    Y = Ishigami.evaluate(param_values)

    Si = delta.analyze(problem, param_values, Y, num_resamples=10,
                       conf_level=0.95, print_to_console=True)

    assert_allclose(Si['delta'], [0.210, 0.358, 0.155], atol=5e-2, rtol=1e-1)
    assert_allclose(Si['S1'], [0.31, 0.44, 0.00], atol=5e-2, rtol=1e-1)
