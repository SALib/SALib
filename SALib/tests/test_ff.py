'''
Created on 30 Jun 2015

@author: will2
'''
from numpy.testing import assert_equal
from SALib.sample import ff
import numpy as np
from scipy.linalg import hadamard

def test_ff_sample():
    problem = {'bounds': [[0., 1.], [0., 1.], [0., 1.], [0., 1.]],
               'num_vars': 4}
    actual = ff.sample(problem)
    expected = np.array([[ 1, 1, 1, 1],
                         [ 1, 0, 1, 0],
                         [ 1, 1, 0, 0],
                         [ 1, 0, 0, 1],
                         [0, 0, 0, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 1],
                         [0, 1, 1, 0]])
    assert_equal(actual, expected)


def test_ff_sample_scaled():
    problem = {'bounds': [[0., 2.5], [0., 1.], [0., 1.], [0., 1.]],
               'num_vars': 4}
    actual = ff.sample(problem)
    expected = np.array([[ 2.5, 1, 1, 1],
                         [ 2.5, 0, 1, 0],
                         [ 2.5, 1, 0, 0],
                         [ 2.5, 0, 0, 1],
                         [0, 0, 0, 0],
                         [0, 1, 0, 1],
                         [0, 0, 1, 1],
                         [0, 1, 1, 0]], dtype=np.float)
    assert_equal(actual, expected)
