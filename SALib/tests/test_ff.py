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
    expected = None
    assert_equal(actual, expected)
