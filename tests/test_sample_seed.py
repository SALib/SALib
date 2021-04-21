import numpy as np

from SALib.sample import fast_sampler, finite_diff, latin, saltelli
from SALib.sample.morris import sample as morris_sampler


def problem_setup():
    N=1

    problem = {'num_vars': 3,
        'names': ['x1','x2','x3'],
        'bounds': [
            [0,1],
            [0,1],
            [0,1]
        ]
    }

    return N, problem


def test_morris_sample_seed():

    N, problem = problem_setup()

    sample1 = morris_sampler(problem, N, seed=None)
    sample2 = morris_sampler(problem, N, seed=123)

    np.testing.assert_equal(np.any(np.not_equal(sample1,sample2)), True)


def test_saltelli_sample_seed():

    N, problem = problem_setup()

    sample1 = saltelli.sample(problem, N, calc_second_order=False, skip_values=1000, check_conv=False)
    sample2 = saltelli.sample(problem, N, calc_second_order=False, skip_values=1001, check_conv=False)

    np.testing.assert_equal(np.any(np.not_equal(sample1,sample2)), True)


def test_fast_sample_seed():

    _, problem = problem_setup()

    sample1 = fast_sampler.sample(problem, 65, seed=None)
    sample2 = fast_sampler.sample(problem, 65, seed=123)

    np.testing.assert_equal(np.any(np.not_equal(sample1,sample2)), True)


def test_finite_diff_sample_seed():
    N, problem = problem_setup()

    sample1 = finite_diff.sample(problem, N, skip_values=1001)
    sample2 = finite_diff.sample(problem, N, skip_values=1002)

    np.testing.assert_equal(np.any(np.not_equal(sample1,sample2)), True)


def test_latin_sample_seed():
    N, problem = problem_setup()

    sample1 = latin.sample(problem, N, seed=None)
    sample2 = latin.sample(problem, N, seed=123)

    np.testing.assert_equal(np.any(np.not_equal(sample1,sample2)), True)

