'''
Created on 30 Jun 2015

@author: will2
'''
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from SALib.sample.ff import sample, find_smallest, extend_bounds
from SALib.analyze.ff import analyze, interactions


def test_find_smallest():
    '''
    '''
    num_vars = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 31, 32, 33]
    expected = [0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 6]
    for x, y in zip(num_vars, expected):
        actual = find_smallest(x)
        assert_equal(actual, y)


def test_extend_bounds():
    problem = {'bounds': np.repeat([-1, 1], 12).reshape(2, 12).T,
           'num_vars': 12,
           'names': ["x" + str(x + 1) for x in range(12)]
           }
    actual = extend_bounds(problem)
    expected = {'names': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'dummy_0', 'dummy_1', 'dummy_2', 'dummy_3'],
                'bounds': [np.array([-1,  1]), np.array([-1,  1]),
                           np.array([-1,  1]), np.array([-1,  1]),
                           np.array([-1,  1]), np.array([-1,  1]),
                           np.array([-1,  1]), np.array([-1,  1]),
                           np.array([-1,  1]), np.array([-1,  1]),
                           np.array([-1,  1]), np.array([-1,  1]),
                           np.array([0, 1]), np.array([0, 1]),
                           np.array([0, 1]), np.array([0, 1])],
                'num_vars': 16}

    assert_equal(actual, expected)

def test_ff_sample():
    problem = {'bounds': [[0., 1.], [0., 1.], [0., 1.], [0., 1.]],
               'num_vars': 4,
               'names': ['x1', 'x2', 'x3', 'x4']}
    actual = sample(problem)
    expected = np.array([[ 1, 1, 1, 1],
                         [ 1, 0, 1, 0],
                         [ 1, 1, 0, 0],
                         [ 1, 0, 0, 1],
                         [0, 0, 0, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 1],
                         [0, 1, 1, 0]], dtype=np.float)
    assert_equal(actual, expected)


def test_ff_sample_scaled():
    '''
    '''
    problem = {'bounds': [[0., 2.5], [0., 1.], [0., 1.], [0., 1.]],
               'num_vars': 4,
               'names': ['x1', 'x2', 'x3', 'x4']}
    actual = sample(problem)
    expected = np.array([[ 2.5, 1, 1, 1],
                         [ 2.5, 0, 1, 0],
                         [ 2.5, 1, 0, 0],
                         [ 2.5, 0, 0, 1],
                         [0, 0, 0, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 1],
                         [0, 1, 1, 0]], dtype=np.float)
    assert_equal(actual, expected)


def test_ff_analyze():
    '''
    '''

    problem = {'bounds': [[0., 2.5], [0., 1.], [0., 1.], [0., 1.]],
               'num_vars': 4,
               'names': ['x1', 'x2', 'x3', 'x4']}
    X = np.array([[ 1, 1, 1, 1],
                  [ 1, 0, 1, 0],
                  [ 1, 1, 0, 0],
                  [ 1, 0, 0, 1],
                  [0, 0, 0, 0],
                  [0, 1, 0, 1],
                  [0, 0, 1, 1],
                  [0, 1, 1, 0]], dtype=np.float)
    Y = np.array([1.5, 1, 1.5, 1, 2, 2.5, 2, 2.5], dtype=np.float)
    actual = analyze(problem, X, Y)
    expected = {'ME': np.array([ -0.5 ,  0.25,  0.  ,  0.  ]), 'names': ['x1', 'x2', 'x3', 'x4']}
    assert_equal(actual, expected)


def test_ff_example():
    '''
    '''

    problem = {'bounds': np.repeat([-1, 1], 12).reshape(2, 12).T,
               'num_vars': 12,
               'names': ["x" + str(x + 1) for x in range(12)]
               }

    X = sample(problem)
    Y = X[:, 0] + 2 * X[:, 1] + 3 * X[:, 2] + 4 * X[:, 6] * X[:, 11]

    expected = np.array([10, -2, 4, -8, 2, 6, -4,
                         0, 2, 6, -4, 0, 10, -2, 4, -8,
                         - 2, -6, 4, 0, -10, 2, -4, 8,
                          - 10, 2, -4, 8, -2, -6, 4, 0])

    assert_equal(Y, expected)

    Si = analyze(problem, X, Y)

    expected = np.array([1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float)
    assert_equal(expected, Si['ME'])


def test_interactions_from_saltelli():
    '''
    '''
    problem = {'bounds': np.repeat([-1, 1], 12).reshape(2, 12).T,
               'num_vars': 12,
               'names': ["x" + str(x + 1) for x in range(12)]
               }

    X = sample(problem)

    Y = np.array([10, -2, 4, -8, 2, 6, -4, 0,
                   2, 6, -4, 0, 10, -2, 4, -8,
                  - 2, -6, 4, 0, -10, 2, -4, 8,
                 - 10, 2, -4, 8, -2, -6, 4, 0])

    Si = analyze(problem, X, Y, second_order=True)
    actual = Si['IE']
    expected = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    assert_equal(actual, expected)


def test_interactions():
    '''
    '''
    problem = {'bounds': [[0., 2.5], [0., 1.], [0., 1.], [0., 1.]],
               'num_vars': 4,
               'names': ['x1', 'x2', 'x3', 'x4']}
    X = np.array([[ 2.5, 1.0, 1.0, 1.0],
                  [ 2.5, 0, 1.0, 0],
                  [ 2.5, 1.0, 0, 0],
                  [ 2.5, 0, 0, 1.0],
                  [0, 0, 0, 0],
                  [0, 1.0, 0, 1.0],
                  [0, 0, 1.0, 1.0],
                  [0, 1.0, 1.0, 0]], dtype=np.float)
    Y = X[:, 0] + (0.1 * X[:, 1]) + ((1.2 * X[:, 2]) * (1.3 + X[:, 3]))
#     Y = np.array([1.5, 1, 1.5, 1, 2, 2.5, 2, 2.5], dtype=np.float)
    ie_names, ie = interactions(problem, Y, print_to_console=True)
    actual = ie
    assert_allclose(actual, [0.3, 0, 0, 0, 0, 0.3], rtol=1e-4, atol=1e-4)
