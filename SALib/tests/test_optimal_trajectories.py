from ..sample.optimal_trajectories import return_max_combo, \
                                          optimised_trajectories
from ..sample.morris_optimal import find_optimum_combination, \
                                    find_optimum_trajectories
from ..util import read_param_file
from nose.tools import raises, with_setup
from numpy.testing import assert_equal
from .test_util import setup_function
from ..sample.morris import Morris
from ..util import read_param_file


@with_setup(setup_function())
def test_optimal_combinations():
    """
    Tests that the optimisation problem gives
    the same answer as the brute force problem
    (for small values of `k_choices` and `N`)
    """
    N = 6
    param_file = "SALib/tests/test_params.txt"
    num_params = pf['num_vars']
    p_levels = 4
    grid_step = p_levels / 2
    k_choices = 4
    morris_sample = Morris(param_file, N, p_levels, grid_step)
    input_sample = morris_sample.get_input_sample_unscaled()

    actual = return_max_combo(input_sample,
                              N,
                              param_file,
                              p_levels,
                              grid_step,
                              k_choices)

    desired = find_optimum_combination(input_sample,
                                        N,
                                        num_params,
                                        k_choices)
    assert_equal(actual, desired)



@with_setup(setup_function())
def test_optimised_trajectories():
    N = 6
    param_file = "SALib/tests/test_params.txt"
    pf = read_param_file(param_file)
    num_params = pf['num_vars']
    p_levels = 4
    grid_step = p_levels / 2
    k_choices = 4
    morris_sample = Morris(param_file, N, p_levels, grid_step)
    input_sample = morris_sample.get_input_sample_unscaled()

    actual = optimised_trajectories(input_sample,
                                    N,
                                    param_file,
                                    p_levels,
                                    grid_step,
                                    k_choices)

    desired = find_optimum_trajectories(input_sample,
                                         N,
                                         num_params,
                                         k_choices)

    assert_equal(actual, desired)


@with_setup(setup_function())
@raises(ValueError)
def test_raise_error_if_k_gt_N():
    """
    Check that an error is raised if `k_choices` is greater than (or equal to) `N`
    """
    N = 4
    param_file = "SALib/tests/test_params.txt"
    p_levels = 4
    grid_step = p_levels / 2
    k_choices = 6
    morris_sample = Morris(param_file, N, p_levels, grid_step)
    input_sample = morris_sample.get_input_sample_unscaled()

    optimised_trajectories(input_sample,
                                    N,
                                    param_file,
                                    p_levels,
                                    grid_step,
                                    k_choices)