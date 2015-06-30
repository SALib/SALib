'''
Created on 30 Jun 2015

@author: will2
'''
from numpy.testing import assert_equal
from SALib.sample.ff import sample
import numpy as np

from SALib.analyze.ff import analyze

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
    expected = {'ME': np.array([ 0.625, 1.   , 0.875, 0.875]), 'names': ['x1', 'x2', 'x3', 'x4']}
    assert_equal(actual, expected)


def test_ff_example():
    '''
    '''

    problem = {'bounds': np.repeat([-1, 1], 12).reshape(2, 12).T,
               'num_vars': 12,
               'names': ["x" + str(x + 1) for x in range(12)]
               }
    
    X = sample(problem)
    print X
    Y = X[:, 0] + 2 * X[:, 1] + 3 * X[:, 2] + 4 * X[:, 6] * X[:, 11]

    expected = np.array([10, -2, 4, -8, 2, 6, -4,
                         0, 2, 6, -4, 0, 10, -2, 4, -8,
                         - 2, -6, 4, 0, -10, 2, -4, 8,
                          - 10, 2, -4, 8, -2, -6, 4, 0])
    
    assert_equal(Y, expected)
    
    Si = analyze(problem, X, Y)

    expected = np.array([1, 2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float)
    assert_equal(expected, Si['ME'])
    
def test_interactions():
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
    actual = analyze(problem, X, Y, second_order=True)
    assert_equal(1,0)