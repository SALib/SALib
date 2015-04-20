from ..sample.optimal_trajectories import return_max_combo
                                          
from ..sample.morris import sample_oat, \
                            find_optimum_combination, \
                            compute_optimised_trajectories, \
                            sample_groups
from . test_morris import setup_param_file_with_groups_prime
from ..util import read_param_file
from nose.tools import raises, with_setup
from numpy.testing import assert_equal
from .test_util import setup_function


@with_setup(setup_param_file_with_groups_prime)
def test_optimal_sample_with_groups():
    '''
    Tests that the combinatorial optimisation approach matches
    that of the brute force approach
    '''
    param_file = "SALib/tests/test_param_file_w_groups_prime.txt"
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

    actual = return_max_combo(sample,
                              N,
                              num_params,
                              k_choices)

    desired = find_optimum_combination(sample,
                                        N,
                                        num_params,
                                        k_choices)

    assert_equal(actual, desired)


@with_setup(setup_param_file_with_groups_prime)
def test_size_of_trajectories_with_groups():
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
    param_file = "SALib/tests/test_param_file_w_groups_prime.txt"
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


    assert_equal(size_x, N*(num_params + 1))
    assert_equal(size_y, num_params)

    # Test 2. dimensions of sample with groups

    group_sample = sample_groups(group_problem, 
                                 N, 
                                 num_levels,
                                 grid_jump)

    size_x, size_y = group_sample.shape

    assert_equal(size_x, N*(num_groups+1))
    assert_equal(size_y, num_params)

    # Test 3. dimensions of optimal sample without groups
    
    optimal_sample_without_groups = compute_optimised_trajectories(no_group_problem, 
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


@with_setup(setup_function())
def test_optimal_combinations():

    N = 6
    param_file = "SALib/tests/test_params.txt"
    problem = read_param_file(param_file)
    num_params = problem['num_vars']
    num_levels = 10
    grid_jump = num_levels / 2
    k_choices = 4

    morris_sample = sample_oat(problem, N, num_levels, grid_jump)

    actual = return_max_combo(morris_sample,
                              N,
                              num_params,
                              k_choices)

    desired = find_optimum_combination(morris_sample,
                                        N,
                                        num_params,
                                        k_choices)

    assert_equal(actual, desired)


@with_setup(setup_function())
def test_optimised_trajectories_without_groups():
    """
    Tests that the optimisation problem gives
    the same answer as the brute force problem
    (for small values of `k_choices` and `N`)
    """
    
    N = 6
    param_file = "SALib/tests/test_params.txt"
    problem = read_param_file(param_file)
    num_levels = 4
    grid_jump = num_levels / 2
    k_choices = 4

    num_params = problem['num_vars']
    groups = problem['groups']

    input_sample = sample_oat(problem, N, num_levels, grid_jump)

    # From gurobi optimal trajectories     
    actual = return_max_combo(input_sample, 
                              N, 
                              num_params, 
                              k_choices, 
                              groups)

    desired = find_optimum_combination(input_sample, 
                                       N, 
                                       num_params, 
                                       k_choices,
                                       groups)
    assert_equal(actual, desired)


@with_setup(setup_param_file_with_groups_prime)
def test_optimised_trajectories_with_groups():
    """
    Tests that the optimisation problem gives
    the same answer as the brute force problem
    (for small values of `k_choices` and `N`)
    with groups
    """
    
    N = 11
    param_file = "SALib/tests/test_param_file_w_groups_prime.txt"
    problem = read_param_file(param_file)
    num_levels = 4
    grid_jump = num_levels / 2
    k_choices = 4
    
    num_params = problem['num_vars']
    groups = problem['groups']

    input_sample = sample_groups(problem, N, num_levels, grid_jump)

    # From gurobi optimal trajectories     
    actual = return_max_combo(input_sample, 
                              N, 
                              num_params, 
                              k_choices, 
                              groups)

    desired = find_optimum_combination(input_sample, 
                                       N, 
                                       num_params, 
                                       k_choices,
                                       groups)
    assert_equal(actual, desired)


@with_setup(setup_function())
@raises(ValueError)
def test_raise_error_if_k_gt_N():
    """
    Check that an error is raised if `k_choices` is greater than (or equal to) `N`
    """
    N = 4
    param_file = "SALib/tests/test_params.txt"
    problem = read_param_file(param_file)
    num_levels = 4
    grid_jump = num_levels / 2
    k_choices = 6

    morris_sample = sample_oat(problem, N, num_levels, grid_jump)


    compute_optimised_trajectories(problem,
                                   morris_sample,
                                   N,
                                   k_choices)
