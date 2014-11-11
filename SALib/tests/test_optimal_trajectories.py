from ..sample.optimal_trajectories import return_max_combo, \
                                          optimised_trajectories
from ..sample.morris_oat import sample
from ..sample.morris_optimal import find_optimum_combination, \
                                    find_optimum_trajectories
from numpy.testing import assert_equal
from ..util import read_param_file


def test_optimal_combinations():
    """
    Tests that the optimisation problem gives
    the same answer as the brute force problem
    (for small values of `k_choices` and `N`)
    """
    N = 6
    param_file = "SALib/tests/test_params.txt"
    pf = read_param_file(param_file)
    num_params = pf['num_vars']
    p_levels = 4
    grid_step = p_levels / 2
    k_choices = 4
    input_sample = sample(N,
                          param_file,
                          num_levels=p_levels,
                          grid_jump=grid_step)

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


def test_optimised_trajectories():
    N = 6
    param_file = "SALib/tests/test_params.txt"
    pf = read_param_file(param_file)
    num_params = pf['num_vars']
    p_levels = 4
    grid_step = p_levels / 2
    k_choices = 4
    input_sample = sample(N,
                          param_file,
                          num_levels=p_levels,
                          grid_jump=grid_step)

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
