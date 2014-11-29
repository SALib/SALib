from __future__ import division
from numpy.testing import assert_equal, assert_allclose
from nose.tools import raises, with_setup
from .test_util import setup_group_file, teardown
import numpy as np
import sys

from ..sample.morris import Morris
from ..analyze import morris
from ..test_functions import Ishigami


@with_setup(setup_group_file, teardown)
def test_regression_morris_vanilla():

    param_file = 'SALib/test_functions/params/Ishigami.txt'
    param_values = Morris(param_file, samples=5000, num_levels=10, grid_jump=5, \
                          group_file=None, \
                          optimal_trajectories=None)

    param_values.save_data('model_input.txt')

    online_model_values = param_values.get_input_sample_scaled()

    Y = Ishigami.evaluate(online_model_values)
    np.savetxt("model_output.txt", Y, delimiter=' ')

    Si = morris.analyze(param_file, 'model_input.txt', 'model_output.txt',
                        column=0, conf_level=0.95, print_to_console=False)

    assert_allclose(Si['mu_star'], [8.1, 2.2, 5.4], atol=0, rtol=5e-1)


@with_setup(setup_group_file, teardown)
def test_regression_morris_groups():

    param_file = 'SALib/test_functions/params/Ishigami.txt'
    group_file = 'SALib/test_functions/groups/Ishigami.txt'

    param_values = Morris(param_file, samples=5000, num_levels=10, grid_jump=5, \
                          group_file=group_file, \
                          optimal_trajectories=None)

    param_values.save_data('model_input_groups.txt')

    online_model_values = param_values.get_input_sample_scaled()

    Y = Ishigami.evaluate(online_model_values)
    np.savetxt("model_output_groups.txt", Y, delimiter=' ')

    Si = morris.analyze(param_file, 'model_input_groups.txt', 'model_output_groups.txt',
                        column=0, conf_level=0.95, print_to_console=False)

    assert_allclose(Si['mu_star'], [8.1, 2.2, 5.4], rtol=5e-1)


@with_setup(setup_group_file, teardown)
def test_regression_morris_optimal():
    '''
    Tests the use of optimal trajectories with Morris.

    Note that the relative tolerance is set to a very high value (default is 1e-05)
    due to the coarse nature of the num_levels and grid_jump.
    '''
    param_file = 'SALib/test_functions/params/Ishigami.txt'
    param_values = Morris(param_file, samples=20, num_levels=4, grid_jump=2, \
                          group_file=None, \
                          optimal_trajectories=9)
    param_values.save_data('model_input_groups.txt')
    online_model_values = param_values.get_input_sample_scaled()
    Y = Ishigami.evaluate(online_model_values)
    np.savetxt("model_output_groups.txt", Y, delimiter=' ')
    Si = morris.analyze(param_file, 'model_input_groups.txt', 'model_output_groups.txt',
                        column=0, conf_level=0.95, print_to_console=False)
    assert_allclose(Si['mu_star'], [8.1, 2.2, 5.4], rtol=10)
