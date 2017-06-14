"""
"""
from SALib.sample.morris import sample_groups
from SALib.sample.morris_strategies import (SampleMorris)
from SALib.sample.morris_strategies.local import LocalOptimisation
from SALib.sample.morris_strategies.brute import BruteForce

from SALib.util import read_param_file, compute_groups_matrix

import numpy as np
from numpy.testing import assert_equal, assert_allclose
from pytest import fixture, raises
import pytest


@fixture(scope='function')
def setup_input():
    input_2 = [[0, 1 / 3.], [2 / 3., 1 / 3.], [2 / 3., 1.]]
    input_1 = [[0, 1 / 3.], [0, 1.], [2 / 3., 1.]]
    input_3 = [[2 / 3., 0], [2 / 3., 2 / 3.], [0, 2 / 3.]]
    input_4 = [[1 / 3., 1.], [1., 1.], [1, 1 / 3.]]
    input_5 = [[1 / 3., 1.], [1 / 3., 1 / 3.], [1, 1 / 3.]]
    input_6 = [[1 / 3., 2 / 3.], [1 / 3., 0], [1., 0]]
    return np.concatenate([input_1, input_2, input_3, input_4, input_5,
                           input_6])


@fixture(scope='function')
def setup_problem():

    input_1 = [[0, 1 / 3.], [0, 1.], [2 / 3., 1.]]
    input_2 = [[0, 1 / 3.], [2 / 3., 1 / 3.], [2 / 3., 1.]]
    input_3 = [[2 / 3., 0], [2 / 3., 2 / 3.], [0, 2 / 3.]]
    input_4 = [[1 / 3., 1.], [1., 1.], [1, 1 / 3.]]
    input_5 = [[1 / 3., 1.], [1 / 3., 1 / 3.], [1, 1 / 3.]]
    input_6 = [[1 / 3., 2 / 3.], [1 / 3., 0], [1., 0]]
    input_sample = np.concatenate([input_1, input_2, input_3,
                                   input_4, input_5, input_6])
    num_samples = 6
    problem = {'num_vars': 2, 'groups': None}
    k_choices = 4

    groups = None
    num_params = problem.get('num_vars')

    expected = np.concatenate([input_1, input_3, input_4, input_6])

    return (input_sample, num_samples, problem,
            k_choices, groups, num_params, expected)


@fixture(scope='function')
def strategy():
    return BruteForce()


class TestSharedMethods:

    def test_check_input_sample_N(self, strategy, setup_input):
        input_sample = setup_input
        num_params = 4
        N = 5
        with raises(AssertionError):
            strategy.check_input_sample(input_sample, num_params, N)

    def test_check_input_sample_num_vars(self, strategy, setup_input):
        input_sample = setup_input
        num_params = 3
        N = 6
        with raises(AssertionError):
            strategy.check_input_sample(input_sample, num_params, N)

    def test_check_input_sample_range(self, strategy, setup_input):
        input_sample = setup_input
        input_sample *= 100
        num_params = 4
        N = 6
        with raises(AssertionError):
            strategy.check_input_sample(input_sample, num_params, N)

    def test_find_maximum(self, strategy):
        scores = np.array(range(15))
        k_choices = 4
        N = 6
        output = strategy.find_maximum(scores, N, k_choices)
        expected = [2, 3, 4, 5]
        assert_equal(output, expected)

    def test_distance(self, strategy):
        '''
        Tests the computation of the distance of two trajectories
        '''
        input_1 = np.matrix(
            [[0, 1 / 3.], [0, 1.], [2 / 3., 1.]], dtype=np.float32)
        input_3 = np.matrix([[2 / 3., 0], [2 / 3., 2 / 3.],
                             [0, 2 / 3.]], dtype=np.float32)
        output = strategy.compute_distance(input_1, input_3)
        assert_allclose(output, 6.18, atol=1e-2)

    def test_distance_of_identical_matrices_is_min(self, strategy):
        input_1 = np.matrix([[1., 1.],
                             [1., 0.33333333],
                             [0.33333333, 0.33333333]])
        input_2 = input_1.copy()
        actual = strategy.compute_distance(input_1, input_2)
        desired = 0
        assert_allclose(actual, desired, atol=1e-2)

    def test_distance_fail_with_difference_size_ip(self, strategy):
        input_1 = np.matrix([[0, 1 / 3.], [0, 1.]], dtype=np.float32)
        input_3 = np.matrix([[2 / 3., 0], [2 / 3., 2 / 3.],
                             [0, 2 / 3.]], dtype=np.float32)
        try:
            strategy.compute_distance(input_1, input_3, 2)
        except:
            pass
        else:
            raise AssertionError(
                "Different size matrices did not trigger error")

    def test_compute_distance_matrix(self, strategy, setup_input):
        '''
        Tests that a distance matrix is computed correctly

        for an input of six trajectories and two parameters
        '''
        sample_inputs = setup_input
        output = strategy.compute_distance_matrix(sample_inputs, 6, 2)
        expected = np.zeros((6, 6), dtype=np.float32)
        expected[1, :] = [5.50, 0, 0, 0, 0, 0]
        expected[2, :] = [6.18, 5.31, 0, 0, 0, 0]
        expected[3, :] = [6.89, 6.18, 6.57, 0, 0, 0]
        expected[4, :] = [6.18, 5.31, 5.41, 5.5, 0, 0]
        expected[5, :] = [7.52, 5.99, 5.52, 7.31, 5.77, 0]
        assert_allclose(output, expected, rtol=1e-2)

    def test_compute_distance_matrix_local(self, strategy, setup_input):
        '''
        Tests that a distance matrix is computed correctly for the
        local distance optimization.
        The only change is that the local method needs the upper triangle of
        the distance matrix instead of the lower one.

        This is for an input of six trajectories and two parameters
        '''
        sample_inputs = setup_input
        output = strategy.compute_distance_matrix(
            sample_inputs, 6, 2, local_optimization=True)
        expected = np.zeros((6, 6), dtype=np.float32)
        expected[0, :] = [0,    5.50, 6.18, 6.89, 6.18, 7.52]
        expected[1, :] = [5.50, 0,    5.31, 6.18, 5.31, 5.99]
        expected[2, :] = [6.18, 5.31, 0,    6.57, 5.41, 5.52]
        expected[3, :] = [6.89, 6.18, 6.57, 0,    5.50, 7.31]
        expected[4, :] = [6.18, 5.31, 5.41, 5.5,  0,    5.77]
        expected[5, :] = [7.52, 5.99, 5.52, 7.31, 5.77, 0]
        assert_allclose(output, expected, rtol=1e-2)


class TestLocallyOptimalStrategy:

    @pytest.mark.xfail
    def test_local(self, setup_problem):
        (input_sample, num_samples, _,
         k_choices, groups, num_params, expected) = setup_problem

        strategy = LocalOptimisation()
        context = SampleMorris(strategy)
        actual = context.sample(input_sample, num_samples, num_params,
                                k_choices, groups)

        np.testing.assert_equal(actual, expected)

    def test_find_local_maximum_distance(self, setup_input):
        '''
        Test whether finding the local maximum distance equals the global
        maximum distance in a simple case.
        From Saltelli et al. 2008, in the solution to exercise 3a,
        Chapter 3, page 134.
        '''

        local_strategy = LocalOptimisation()
        brute_strategy = BruteForce()

        sample_inputs = setup_input
        N = 6
        num_params = 2
        k_choices = 4
        scores_global = brute_strategy.find_most_distant(sample_inputs, N,
                                                         num_params, k_choices)
        output_global = brute_strategy.find_maximum(
            scores_global, N, k_choices)
        output_local = local_strategy.find_local_maximum(sample_inputs, N,
                                                         num_params, k_choices)
        assert_equal(output_global, output_local)

    @pytest.mark.xfail
    def test_local_optimised_groups(self,
                                    setup_param_groups_prime):
        """
        Tests that the local optimisation problem gives
        the same answer as the brute force problem
        (for small values of `k_choices` and `N`)
        with groups
        """
        N = 8
        param_file = setup_param_groups_prime
        problem = read_param_file(param_file)
        num_levels = 4
        grid_jump = num_levels / 2
        k_choices = 4

        num_params = problem['num_vars']
        groups = problem['groups']

        input_sample = sample_groups(problem, N, num_levels, grid_jump)

        groups = compute_groups_matrix(groups, num_params)

        strategy = LocalOptimisation()

        # From local optimal trajectories
        actual = strategy.find_local_maximum(input_sample, N, num_params,
                                             k_choices, groups)

        desired = strategy.locally_optimal_combination(input_sample,
                                                       N,
                                                       num_params,
                                                       k_choices,
                                                       groups)
        assert_equal(actual, desired)


class TestLocalMethods:

    def test_sum_distances(self, setup_input):
        '''
        Tests whether the combinations are summed correctly.
        '''
        strategy = LocalOptimisation()

        sample_inputs = setup_input
        dist_matr = strategy.compute_distance_matrix(sample_inputs, 6, 2,
                                                     local_optimization=True)
        indices = (1, 3, 2)
        distance = strategy.sum_distances(indices, dist_matr)

        expected = 10.47
        assert_allclose(distance, expected, rtol=1e-2)

    def test_get_max_sum_ind(self):
        '''
        Tests whether the right maximum indices are returned.
        '''
        strategy = LocalOptimisation()

        indices = np.array([(1, 2, 4), (3, 2, 1), (4, 2, 1)])
        distances = np.array([20, 40, 50])

        output = strategy.get_max_sum_ind(indices, distances, 0, 0)
        expected = (4, 2, 1)

        assert_equal(output, expected)

    def test_add_indices(self):
        '''
        Tests whether the right indices are added.
        '''
        strategy = LocalOptimisation()

        indices = (1, 3, 4)
        matr = np.zeros((6, 6), dtype=np.int16)
        ind_extra = strategy.add_indices(indices, matr)

        expected = [(1, 3, 4, 0), (1, 3, 4, 2), (1, 3, 4, 5)]

        assert_equal(ind_extra, expected)

    def test_get_max_sum_index_raises_error(self):
        strategy = LocalOptimisation()
        indices = [(1, 2, 4), (3, 2, 1), (4, 2, 1)]
        distances_wrong = [20, 40]

        with raises(ValueError):
            strategy.get_max_sum_ind(indices, distances_wrong, 0, 0)


class TestBruteForceStrategy:

    def test_brute_force(self, setup_problem):

        (input_sample, num_samples, _,
         k_choices, groups, num_params, expected) = setup_problem

        strategy = BruteForce()
        context = SampleMorris(strategy)
        actual = context.sample(input_sample, num_samples, num_params,
                                k_choices, groups)

        np.testing.assert_equal(actual, expected)


class TestBruteForceMethods:

    def test_combo_from_find_most_distant(self, setup_input):
        '''
        Tests whether the correct combination is picked from the fixture drawn
        from Saltelli et al. 2008, in the solution to exercise 3a,
        Chapter 3, page 134.
        '''
        sample_inputs = setup_input
        N = 6
        num_params = 2
        k_choices = 4
        strategy = BruteForce()
        scores = strategy.find_most_distant(sample_inputs, N, num_params,
                                            k_choices)
        output = strategy.find_maximum(scores, N, k_choices)
        expected = [0, 2, 3, 5]  # trajectories 1, 3, 4, 6
        assert_equal(output, expected)

    def test_scores_from_find_most_distant(self, setup_input):
        '''
        Checks whether array of scores from (6 4) is correct.

        Data is derived from Saltelli et al. 2008,
        in the solution to exercise 3a, Chapter 3, page 134.

        '''
        sample_inputs = setup_input
        N = 6
        num_params = 2
        k_choices = 4
        strategy = BruteForce()
        output = strategy.find_most_distant(sample_inputs, N, num_params,
                                            k_choices)
        expected = np.array([15.022, 13.871, 14.815, 14.582, 16.178, 14.912,
                             15.055, 16.410, 15.685, 16.098, 14.049, 15.146,
                             14.333, 14.807, 14.825],
                            dtype=np.float32)

        assert_allclose(output, expected, rtol=1e-1, atol=1e-2)

    def test_catch_combos_too_large(self):
        N = int(1e6)
        k_choices = 4
        num_params = 2
        input_sample = np.random.random_sample((N, num_params))
        strategy = BruteForce()
        with raises(ValueError):
            strategy.find_most_distant(input_sample, N, num_params, k_choices)

    def test_make_index_list(self):
        N = 4
        num_params = 2
        groups = None
        strategy = BruteForce()
        actual = strategy._make_index_list(N, num_params, groups)
        desired = [np.array([0, 1, 2]), np.array([3, 4, 5]),
                   np.array([6, 7, 8]), np.array([9, 10, 11])]
        assert_equal(desired, actual)

    def test_make_index_list_with_groups(self):
        N = 4
        num_params = 3
        groups = 2
        strategy = BruteForce()
        actual = strategy._make_index_list(N, num_params, groups)
        desired = [np.array([0, 1, 2]), np.array([3, 4, 5]),
                   np.array([6, 7, 8]), np.array([9, 10, 11])]
        assert_equal(desired, actual)
