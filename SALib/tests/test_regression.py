from __future__ import division
from numpy.testing import assert_allclose
from nose.tools import with_setup
from .test_util import teardown
import numpy as np
import sys

from ..sample.morris import sample
from ..analyze import morris
from ..test_functions import Ishigami
from ..util import read_param_file


def test_regression_morris_vanilla():

    param_file ='SALib/test_functions/params/Ishigami.txt'
    problem = read_param_file(param_file)
    param_values = sample(problem=problem, N=5000, \
                          num_levels=10, grid_jump=5, \
                          optimal_trajectories=None)

    np.savetxt('model_input.txt', param_values, delimiter=' ')

    Y = Ishigami.evaluate(param_values)
    np.savetxt("model_output.txt", Y, delimiter=' ')

    Si = morris.analyze(param_file, 'model_input.txt', 'model_output.txt',
                        column=0, conf_level=0.95, print_to_console=False,
                        num_levels=10, grid_jump=5)

    assert_allclose(Si['mu_star'], [8.1, 2.2, 5.4], atol=0, rtol=5e-1)


def test_regression_morris_groups():

    param_file = 'SALib/test_functions/params/Ishigami_groups.txt'
    problem = read_param_file(param_file)

    param_values = sample(problem=problem, N=5000, \
                          num_levels=10, grid_jump=5, \
                          optimal_trajectories=None)

    np.savetxt('model_input_groups.txt', param_values, delimiter=' ')

    Y = Ishigami.evaluate(param_values)
    np.savetxt("model_output_groups.txt", Y, delimiter=' ')

    Si = morris.analyze(param_file, 'model_input_groups.txt', 'model_output_groups.txt',
                        column=0, conf_level=0.95, print_to_console=False,
                        num_levels=10, grid_jump=5)

    assert_allclose(Si['mu_star'], [7.87, 6.26], rtol=5e-1)


def test_regression_morris_optimal():
    '''
    Tests the use of optimal trajectories with Morris.

    Note that the relative tolerance is set to a very high value (default is 1e-05)
    due to the coarse nature of the num_levels and grid_jump.
    '''
    param_file = 'SALib/test_functions/params/Ishigami.txt'
    problem = read_param_file(param_file)
    param_values = sample(problem=problem, N=20, \
                          num_levels=4, grid_jump=2, \
                          optimal_trajectories=9)

    np.savetxt('model_input_groups.txt', param_values, delimiter=' ')

    Y = Ishigami.evaluate(param_values)

    np.savetxt("model_output_groups.txt", Y, delimiter=' ')

    Si = morris.analyze(param_file, 'model_input_groups.txt', 'model_output_groups.txt',
                        column=0, conf_level=0.95, print_to_console=False,
                        num_levels=4, grid_jump=2)

    assert_allclose(Si['mu_star'], [8.1, 2.2, 5.4], rtol=10)
