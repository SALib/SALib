from __future__ import division
from numpy.testing import assert_equal
from ..util.morris_trajectories import generate_P_star,\
                                       compute_B_star, \
                                       compute_delta, \
                                       generate_trajectory
import numpy as np


def test_generate_P_star():
    '''
    Matrix P* - size (g * g) - describes order in which groups move
    each row contains one element equal to 1, all others are 0
    no two columns have 1s in the same position
    '''
    for i in range(1,100):
        output = generate_P_star(i)
        if np.any(np.sum(output,0) != np.ones(i)):
            raise AssertionError("Not legal P along axis 0")
        elif np.any(np.sum(output, 1) != np.ones(i)):
            raise AssertionError("Not legal P along axis 1")


def test_compute_delta():
    fixture = np.arange(2,10)
    output = [compute_delta(f) for f in fixture]
    desired = np.array([1.00, 0.75, 0.66, 0.62,
                        0.60, 0.58, 0.57, 0.56])
    np.testing.assert_almost_equal(output, desired, decimal=2)


def test_generate_trajectory():
    # Two groups of three factors
    G = np.array([[1,0],[0,1],[0,1]])
    # Four levels, grid_jump = 2
    num_levels, grid_jump = 4, 2
    output = generate_trajectory(G, num_levels, grid_jump)
    if np.any((output > 1) | (output < 0)):
        raise AssertionError("Bound not working")
    assert_equal(output.shape[0],3)
    assert_equal(output.shape[1],3)


def test_compute_B_star():
    '''
    Tests for expected output

    Taken from example 3.2 in Saltelli et al. (2008) pg 122
    '''

    k = 3
    g = 2

    x_star = np.matrix(np.array([1./3, 1./3, 0.]))
    J = np.matrix(np.ones((g + 1, k)))
    G = np.matrix('1,0;0,1;0,1')
    D_star = np.matrix('1,0,0;0,-1,0;0,0,1')
    P_star = np.matrix('1,0;0,1')
    delta = 2./3
    B = np.matrix(np.tril(np.ones([g + 1, g], dtype=int), -1))

    desired = np.array([[1./3, 1, 0],[1, 1, 0],[1, 1./3, 2./3]])

    output = compute_B_star(J, x_star, delta, B, G, P_star, D_star)
    assert_equal(output, desired)

