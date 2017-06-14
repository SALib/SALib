"""
"""
from SALib.sample.morris import sample_groups
from SALib.sample.morris_strategies import (SampleMorris)
from SALib.sample.morris_strategies.local import LocalOptimisation
from SALib.sample.morris_strategies.brute import BruteForce

from SALib.sample.morris_util import (find_maximum,
                                      find_most_distant,
                                      compute_distance_matrix)

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
        # brute_strategy = BruteForce()

        sample_inputs = setup_input
        N = 6
        num_params = 2
        k_choices = 4
        scores_global = find_most_distant(sample_inputs, N,
                                          num_params, k_choices)
        output_global = find_maximum(scores_global, N, k_choices)
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
        dist_matr = compute_distance_matrix(sample_inputs, 6, 2,
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
