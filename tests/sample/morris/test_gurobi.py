from unittest import skipUnless

import numpy as np
from numpy.testing import assert_equal
from nose.tools import raises

from SALib.sample.morris.brute import BruteForce
from SALib.sample.morris.gurobi import GlobalOptimisation
from SALib.sample.morris.local import LocalOptimisation

from SALib.sample.morris import sample_oat, \
    compute_optimised_trajectories, \
    sample_groups


from SALib.util import read_param_file

try:
    import gurobipy
except ImportError:
    _has_gurobi = False
else:
    _has_gurobi = True


@skipUnless(_has_gurobi, "Gurobi is required for combinatorial optimisation")
def test_optimal_sample_with_groups(setup_param_groups_prime):
    '''
    Tests that the combinatorial optimisation approach matches
    that of the brute force approach
    '''
    param_file = setup_param_groups_prime
    problem = read_param_file(param_file)

    N = 10
    num_levels = 8
    grid_jump = 4
    k_choices = 4
    num_params = problem['num_vars']

    sample = sample_oat(problem,
                        N,
                        num_levels,
                        grid_jump)

    strategy = GlobalOptimisation()
    actual = strategy.return_max_combo(sample,
                                       N,
                                       num_params,
                                       k_choices)

    brute_strategy = BruteForce()
    desired = brute_strategy.brute_force_most_distant(sample,
                                                      N,
                                                      num_params,
                                                      k_choices)

    assert_equal(actual, desired)


@skipUnless(_has_gurobi, "Gurobi is required for combinatorial optimisation")
def test_size_of_trajectories_with_groups(setup_param_groups_prime):
    '''
    Tests that the number of trajectories produced is computed
    correctly (i.e. that the size of the trajectories is a function
    of the number of groups, rather than the number of variables
    when groups are used.

    There are seven variables and three groups.
    With N=10:
    1. the sample ignoring groups (i.e. the call to `sample_oat')
    should be of size N*(D+1)-by-D.
    2. the sample with groups should be of size N*(G+1)-by-D
    When k=4:
    3. the optimal sample ignoring groups should be of size k*(D+1)-by-D
    4. the optimal sample with groups should be of size k*(G+1)-by-D
    '''
    param_file = setup_param_groups_prime
    group_problem = read_param_file(param_file)
    no_group_problem = read_param_file(param_file)
    no_group_problem['groups'] = None

    N = 11
    num_levels = 8
    grid_jump = 4
    k_choices = 4
    num_params = group_problem['num_vars']

    num_groups = 3

    # Test 1. dimensions of sample ignoring groups
    sample = sample_oat(no_group_problem,
                        N,
                        num_levels,
                        grid_jump)

    size_x, size_y = sample.shape

    assert_equal(size_x, N * (num_params + 1))
    assert_equal(size_y, num_params)

    # Test 2. dimensions of sample with groups

    group_sample = sample_groups(group_problem,
                                 N,
                                 num_levels,
                                 grid_jump)

    size_x, size_y = group_sample.shape

    assert_equal(size_x, N * (num_groups + 1))
    assert_equal(size_y, num_params)

    # Test 3. dimensions of optimal sample without groups

    optimal_sample_without_groups = \
        compute_optimised_trajectories(no_group_problem,
                                       sample,
                                       N,
                                       k_choices)

    size_x, size_y = optimal_sample_without_groups.shape

    assert_equal(size_x, k_choices * (num_params + 1))
    assert_equal(size_y, num_params)

    # Test 4. dimensions of optimal sample with groups

    optimal_sample_with_groups = compute_optimised_trajectories(group_problem,
                                                                group_sample,
                                                                N,
                                                                k_choices)

    size_x, size_y = optimal_sample_with_groups.shape

    assert_equal(size_x, k_choices * (num_groups + 1))
    assert_equal(size_y, num_params)


@skipUnless(_has_gurobi, "Gurobi is required for combinatorial optimisation")
def test_optimal_combinations(setup_function):

    N = 6
    param_file = setup_function
    problem = read_param_file(param_file)
    num_params = problem['num_vars']
    num_levels = 10
    grid_jump = num_levels / 2
    k_choices = 4

    morris_sample = sample_oat(problem, N, num_levels, grid_jump)

    global_strategy = GlobalOptimisation()
    actual = global_strategy.return_max_combo(morris_sample,
                                              N,
                                              num_params,
                                              k_choices)

    local_strategy = LocalOptimisation()
    desired = local_strategy.locally_optimal_combination(morris_sample,
                                                         N,
                                                         num_params,
                                                         k_choices)

    assert_equal(actual, desired)


@skipUnless(_has_gurobi, "Gurobi is required for combinatorial optimisation")
def test_optimised_trajectories_without_groups(setup_function):
    """
    Tests that the optimisation problem gives
    the same answer as the brute force problem
    (for small values of `k_choices` and `N`),
    particularly when there are two or more identical
    trajectories
    """

    N = 6
    param_file = setup_function
    problem = read_param_file(param_file)
    k_choices = 4

    num_params = problem['num_vars']
    groups = problem['groups']

    # 6 trajectories, with 5th and 6th identical
    input_sample = np.array([[0.33333333,  0.66666667],
                             [1., 0.66666667],
                             [1., 0.],
                             [0., 0.33333333],
                             [0., 1.],
                             [0.66666667, 1.],
                             [0.66666667, 0.33333333],
                             [0.66666667, 1.],
                             [0., 1.],
                             [0.66666667, 1.],
                             [0.66666667, 0.33333333],
                             [0., 0.33333333],
                             [1., 1.],
                             [1., 0.33333333],
                             [0.33333333, 0.33333333],
                             [1., 1.],
                             [1., 0.33333333],
                             [0.33333333, 0.33333333]], dtype=np.float32)

    print(input_sample)

    # From gurobi optimal trajectories
    strategy = GlobalOptimisation()
    actual = strategy.return_max_combo(input_sample,
                                       N,
                                       num_params,
                                       k_choices)

    local_strategy = LocalOptimisation()
    desired = local_strategy.locally_optimal_combination(input_sample,
                                                         N,
                                                         num_params,
                                                         k_choices,
                                                         groups)

    assert_equal(actual, desired)


@skipUnless(_has_gurobi, "Gurobi is required for combinatorial optimisation")
def test_optimised_trajectories_groups(setup_param_groups_prime):
    """
    Tests that the optimisation problem gives
    the same answer as the brute force problem
    (for small values of `k_choices` and `N`)
    with groups
    """

    N = 11
    param_file = setup_param_groups_prime
    problem = read_param_file(param_file)
    num_levels = 4
    grid_jump = num_levels / 2
    k_choices = 4

    num_params = problem['num_vars']
    groups = problem['groups']

    input_sample = sample_groups(problem, N, num_levels, grid_jump)

    # From gurobi optimal trajectories
    strategy = GlobalOptimisation()
    actual = strategy.return_max_combo(input_sample,
                                       N,
                                       num_params,
                                       k_choices)

    local_strategy = LocalOptimisation()
    desired = local_strategy.locally_optimal_combination(input_sample,
                                                         N,
                                                         num_params,
                                                         k_choices,
                                                         groups)
    assert_equal(actual, desired)


@skipUnless(_has_gurobi, "Gurobi is required for combinatorial optimisation")
@raises(ValueError)
def test_raise_error_if_k_gt_N(setup_function):
    """Check that an error is raised if `k_choices` is greater than
    (or equal to) `N`
    """
    N = 4
    param_file = setup_function
    problem = read_param_file(param_file)
    num_levels = 4
    grid_jump = num_levels / 2
    k_choices = 6

    morris_sample = sample_oat(problem, N, num_levels, grid_jump)

    compute_optimised_trajectories(problem,
                                   morris_sample,
                                   N,
                                   k_choices)
