from __future__ import division

from numpy.testing import assert_equal, assert_allclose

from pytest import raises, fixture, warns, mark

import numpy as np
import warnings

from SALib.sample.morris.morris import (sample,
                                        _check_group_membership,
                                        _check_if_num_levels_is_even,
                                        _compute_optimised_trajectories,
                                        _generate_p_star,
                                        _compute_b_star,
                                        _generate_trajectory,
                                        _generate_x_star)

from SALib.util import (read_param_file, compute_groups_matrix,
                        _define_problem_with_groups, _compute_delta)


@fixture(scope='function')
def setup_input():
    input_1 = [[0, 1 / 3.], [0, 1.], [2 / 3., 1.]]
    input_2 = [[0, 1 / 3.], [2 / 3., 1 / 3.], [2 / 3., 1.]]
    input_3 = [[2 / 3., 0], [2 / 3., 2 / 3.], [0, 2 / 3.]]
    input_4 = [[1 / 3., 1.], [1., 1.], [1, 1 / 3.]]
    input_5 = [[1 / 3., 1.], [1 / 3., 1 / 3.], [1, 1 / 3.]]
    input_6 = [[1 / 3., 2 / 3.], [1 / 3., 0], [1., 0]]
    input_sample = np.concatenate([input_1, input_2, input_3,
                                   input_4, input_5, input_6])
    return input_sample


@fixture(scope='function')
def expected_sample():
    input_1 = [[0, 1 / 3.], [0, 1.], [2 / 3., 1.]]
    input_3 = [[2 / 3., 0], [2 / 3., 2 / 3.], [0, 2 / 3.]]
    input_4 = [[1 / 3., 1.], [1., 1.], [1, 1 / 3.]]
    input_6 = [[1 / 3., 2 / 3.], [1 / 3., 0], [1., 0]]
    return np.concatenate([input_1, input_3, input_4, input_6])


def test_odd_num_levels_raises_warning(setup_param_file_with_groups):
    parameter_file = setup_param_file_with_groups
    problem = read_param_file(parameter_file)
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        sample(problem, 10, num_levels=3)
        # Verify some things
        assert len(w) == 1
        assert issubclass(w[-1].category, UserWarning)
        assert "num_levels should be an even number, sample may be biased" in str(
            w[-1].message)


def test_even_num_levels_no_warning(setup_param_file_with_groups):
    parameter_file = setup_param_file_with_groups
    problem = read_param_file(parameter_file)
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        # Trigger a warning.
        sample(problem, 10, num_levels=4)
        # Verify some things
        assert len(w) == 0


def test_group_in_param_file_read(setup_param_file_with_groups):
    '''
    Tests that groups in a parameter file are read correctly
    '''
    parameter_file = setup_param_file_with_groups
    problem = read_param_file(parameter_file)
    groups, group_names = compute_groups_matrix(
        problem['groups'])

    assert_equal(problem['names'], ["Test 1", "Test 2", "Test 3"])
    assert_equal(groups, np.array([[1, 0], [1, 0], [0, 1]], dtype=np.int))
    assert_equal(group_names, ['Group 1', 'Group 2'])


def test_optimal_trajectories_lt_samples(setup_param_file):
    parameter_file = setup_param_file
    problem = read_param_file(parameter_file)

    samples = 10
    num_levels = 4

    with raises(ValueError):
        sample(problem, samples, num_levels,
               optimal_trajectories=samples)


def test_optimal_trajectories_lt_10(setup_param_file):
    parameter_file = setup_param_file
    problem = read_param_file(parameter_file)

    samples = 10
    num_levels = 4

    optimal_trajectories = 11
    with raises(ValueError):
        sample(problem, samples, num_levels,
               optimal_trajectories=optimal_trajectories)


def test_optimal_trajectories_gte_one(setup_param_file):
    parameter_file = setup_param_file
    problem = read_param_file(parameter_file)

    samples = 10
    num_levels = 4
    optimal_trajectories = 1

    with raises(ValueError):
        sample(problem, samples, num_levels,
               optimal_trajectories)


def test_find_optimum_trajectories(setup_input, expected_sample):
    N = 6
    problem = {'num_vars': 2, 'names': ['x1', 'x2'], 'groups': None}
    k_choices = 4

    problem = _define_problem_with_groups(problem)

    output = _compute_optimised_trajectories(
        problem, setup_input, N, k_choices)
    expected = expected_sample
    np.testing.assert_equal(output, expected)


def test_catch_inputs_not_in_zero_one_range(setup_input):
    problem = {'num_vars': 2, 'groups': None}
    k_choices = 4
    N = 10
    with raises(ValueError):
        _compute_optimised_trajectories(problem, setup_input * 10, N,
                                        k_choices)


def test_group_sample_fails_with_wrong_G_matrix():
    N = 6
    num_levels = 4

    problem = {'bounds': [[0., 1.], [0., 1.], [0., 1.], [0., 1.]],
               'num_vars': 4,
               'groups': list([1, 2, 3])}

    with raises(ValueError) as err:
        sample(problem, N, num_levels)

    assert "Number of entries in \'groups\' should be the same as in " \
           "\'names\'" in str(err.value)


class TestGroupSampleGeneration:

    def test_generate_p_star(self):
        '''
        Matrix P* - size (g * g) - describes order in which groups move
        each row contains one element equal to 1, all others are 0
        no two columns have 1s in the same position
        '''
        for i in range(1, 100):
            output = _generate_p_star(i)
            if np.any(np.sum(output, 0) != np.ones(i)):
                raise AssertionError("Not legal P along axis 0")
            elif np.any(np.sum(output, 1) != np.ones(i)):
                raise AssertionError("Not legal P along axis 1")

    def test_compute_delta(self):
        fixture = np.arange(2, 10)
        output = [_compute_delta(f) for f in fixture]
        desired = np.array([1.0, 0.75, 0.666667, 0.625,
                            0.6, 0.583333, 0.571429, 0.5625])
        assert_allclose(output, desired, rtol=1e-2)

    def test_generate_trajectory(self):
        # Two groups of three factors
        G = np.array([[1, 0], [0, 1], [0, 1]])
        # Four levels
        num_levels = 4
        output = _generate_trajectory(G, num_levels)
        if np.any((output > 1) | (output < 0)):
            raise AssertionError("Bound not working: %s", output)
        assert_equal(output.shape[0], 3)
        assert_equal(output.shape[1], 3)

    def test_compute_B_star(self):
        '''
        Tests for expected output

        Taken from example 3.2 in Saltelli et al. (2008) pg 122
        '''

        k = 3
        g = 2

        x_star = np.array([[1. / 3, 1. / 3, 0.]])
        J = np.ones((g + 1, k))
        G = np.array([[1, 0], [0, 1], [0, 1]])
        D_star = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        P_star = np.array([[1, 0], [0, 1]])
        delta = 2. / 3
        B = np.tril(np.ones([g + 1, g], dtype=int), -1)

        desired = np.array([[1. / 3, 1, 0], [1, 1, 0], [1, 1. / 3, 2. / 3]])

        output = _compute_b_star(J, x_star, delta, B, G, P_star, D_star)
        assert_allclose(output, desired)

    def test_generate_x_star(self):
        """
        """
        num_params = 4
        num_levels = 4

        np.random.seed(10)
        actual = _generate_x_star(num_params, num_levels)
        print(actual)
        expected = np.array([[0.333333, 0.333333, 0., 0.333333]])
        assert_allclose(actual, expected, rtol=1e-05)

    def test_define_problem_with_groups_all_ok(self):
        """
        Checks if the function works when the user defines different groups for
        each variable.
        """
        problem = {
            'num_vars': 8,
            'names': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'],
            'groups': ['G1', 'G1', 'G1', 'G2', 'G2', 'G2', 'G3', 'G3']}

        expected = problem

        result = _define_problem_with_groups(problem)

        assert expected == result

    def test_define_problem_with_groups_no_group_definition(self):
        """
        Checks if the function works when the user doesn't define groups at
        all.
        """
        problem = {
            'num_vars': 8,
            'names': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']}

        expected = {
            'num_vars': 8,
            'names': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'],
            'groups': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']}

        result = _define_problem_with_groups(problem)

        assert expected == result

    def test_define_problem_with_groups_exception(self):
        """
        Checks if the function raises an exception when the user makes an
        inconsistent definition of groups, i.e, only define groups for some
        variables.
        """
        problem = {
            'num_vars': 8,
            'names': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8'],
            'groups': ['G1', 'G1', 'G1', 'G2', 'G2', 'G2']}

        with raises(ValueError):
            _define_problem_with_groups(problem)

    def test_check_if_num_levels_is_even_check_odd(self):
        """
        Checks if a warn is raised when the number of tests is odd
        """
        with warns(None) as record:
            _check_if_num_levels_is_even(5)

        assert record

    def test_check_if_num_levels_is_even_check_even(self):
        """
        Checks if a warn is not raised when the number of tests is even.
        """
        with warns(None) as record:
            _check_if_num_levels_is_even(4)

        assert not record

    @mark.xfail()
    def test_check_group_membership_all_ok(self):
        """
        Checks if no errors are raised when group_membership is defined
        correctly. This test is expected to fail.
        """
        # Creates a dummy variable
        group_membership = np.empty((3, 3), dtype=np.int)

        with raises(ValueError):
            _check_group_membership(group_membership)

        with raises(TypeError):
            _check_group_membership(group_membership)

    def test_check_group_membership_error(self):
        """
        Checks if an error is raised if group_membership has a wrong type.
        """
        # Creates a dummy variable
        group_membership = 1.0

        with raises(TypeError):
            _check_group_membership(group_membership)

        group_membership = None

        with raises(ValueError):
            _check_group_membership(group_membership)
