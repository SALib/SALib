from pytest import fixture

import numpy as np
from numpy.testing import assert_allclose

from SALib.analyze import delta, dgsm, fast, rbd_fast, sobol, morris, hdmr
from SALib.sample import fast_sampler, finite_diff, latin, saltelli
from SALib.sample.morris import sample as morris_sampler

from SALib.test_functions import Ishigami
from SALib.test_functions import linear_model_1
from SALib.test_functions import linear_model_2
from SALib.util import read_param_file


@fixture(scope='session')
def set_seed():
    """Sets seeds for random generators so that tests can be repeated

    It is necessary to set seeds for both the numpy.random, and
    the stdlib.random libraries.
    """
    seed = 101
    np.random.seed(seed)


@fixture(scope='function')
def test_problem():
    problem = {
        'names': ['x1', 'x2', 'x3'],
        'num_vars': 3,
        'bounds': [[-np.pi, np.pi], [1.0, 0.2], [3, 0.5]],
        'groups': ['G1', 'G1', 'G2'],
        'dists': ['unif', 'norm', 'triang']
    }

    return problem


def test_sobol_group(set_seed, test_problem):
    X = saltelli.sample(test_problem, 100)
    Y = Ishigami.evaluate(X)
    Si = sobol.analyze(test_problem, Y)

    total, first, second = Si.to_df()
    expected = ['G1', 'G2']

    assert total.index.tolist() == expected
    assert first.index.tolist() == expected
    assert second.index.tolist() == [tuple(expected)]


def test_morris_group(set_seed, test_problem):
    X = morris_sampler(test_problem, 100)
    Y = Ishigami.evaluate(X)
    Si = morris.analyze(test_problem, X, Y)

    df = Si.to_df()
    expected = ['G1', 'G2']

    assert df.index.tolist() == expected


def test_fast_group(set_seed, test_problem):
    X = morris_sampler(test_problem, 100)
    Y = Ishigami.evaluate(X)
    Si = morris.analyze(test_problem, X, Y)

    df = Si.to_df()
    expected = ['G1', 'G2']

    assert df.index.tolist() == expected


def test_rbd_fast_group(set_seed, test_problem):
    X = latin.sample(test_problem, 100)
    Y = Ishigami.evaluate(X)
    Si = rbd_fast.analyze(test_problem, X, Y)

    df = Si.to_df()
    expected = ['G1', 'G2']

    assert df.index.tolist() == expected


def test_delta_group(set_seed, test_problem):
    X = latin.sample(test_problem, 100)
    Y = Ishigami.evaluate(X)
    Si = delta.analyze(test_problem, X, Y)

    df = Si.to_df()
    expected = ['G1', 'G2']

    assert df.index.tolist() == expected


# def test_dgsm_group(set_seed, test_problem):
#     X = finite_diff.sample(test_problem, 1000)
#     Y = Ishigami.evaluate(X)
#     Si = dgsm.analyze(test_problem, X, Y)

#     df = Si.to_df()
#     expected = ['G1', 'G2']

#     assert df.index.tolist() == expected


# def test_hdmr_group(set_seed, test_problem):
#     X = latin.sample(test_problem, 300)
#     Y = Ishigami.evaluate(X)
#     Si = hdmr.analyze(test_problem, X, Y)

#     df = Si.to_df()
#     expected = ['G1', 'G2']

#     assert df.index.tolist() == expected



