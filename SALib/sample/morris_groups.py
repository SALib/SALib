from __future__ import division
import numpy as np
import random as rd


def generate_trajectory(G, num_levels, grid_jump):
    '''
    Returns a single trajectory of size (g+1)-by-k
    where g is the number of groups,
    and k is the number of factors, both implied by the dimensions of G

    Arguments:
      G            a k-by-g matrix which notes factor membership of groups
      num_levels   integer describing number of levels
      grid_jump    recommended to be equal to p / (2(p-1)) where p is num_levels
    '''

    delta = compute_delta(num_levels)

    # Infer number of groups g and number of params k from matrix G
    k = G.shape[0]
    g = G.shape[1]

    # Matrix B - size (g + 1) * g -  lower triangular matrix
    B = np.matrix(np.tril(np.ones([g + 1, g], dtype=int), -1))

    P_star = np.asmatrix(generate_P_star(g))

    # Matrix J - a (g+1)-by-k matrix of ones
    J = np.matrix(np.ones((g+1, k)))

    # Matrix D* - k-by-k matrix which decribes whether factors move up or down
    D_star = np.diag([rd.choice([-1, 1]) for _ in range(k)])

    x_star = np.asmatrix(generate_x_star(k, num_levels, grid_jump))

    # Matrix B* - size (g + 1) * k
    B_star = compute_B_star(J, x_star, delta, B, G, P_star, D_star)

    return B_star


def sample(N, G, num_levels, grid_jump):
    '''
    Returns an N(g+1)-by-k array of N trajectories;
    where g is the number of groups and k is the number of factors

    Arguments:
      N            number of trajectories
      G            a k-by-g matrix which denotes factor membership of groups
      num_levels   integer describing number of levels
      grid_jump    recommended to be equal to p / (2(p-1)) where p is num_levels
    '''

    if G is None:
        raise ValueError("Please define the matrix G.")
    if type(G) is not np.matrixlib.defmatrix.matrix:
       raise TypeError("Matrix G should be formatted as a numpy matrix")

    k = G.shape[0]
    g = G.shape[1]
    sample = np.empty((N*(g + 1), k))
    sample = np.array([generate_trajectory(G, num_levels, grid_jump) for n in range(N)])
    return sample.reshape((N*(g + 1), k))


def compute_B_star(J, x_star, delta, B, G, P_star, D_star):
    B_star = J[:,0] * x_star + \
             (delta / 2) *  (( 2 * B * (G * P_star).T - J) \
             * D_star + J)
    return B_star


def generate_P_star(g):
    '''
    Matrix P* - size (g-by-g) - describes order in which groups move
    '''
    P_star = np.eye(g,g)
    np.random.shuffle(P_star)
    return P_star


def generate_x_star(k, num_levels, grid_step):
    '''
    Generate an 1-by-k array to represent initial position for EE
    This should be a randomly generated array in the p level grid :math:\omega
    '''
    x_star = np.empty(k)

    delta = compute_delta(num_levels)
    bound = 1 - delta
    grid = np.linspace(0,bound,grid_step)

    for i in range(k):
        x_star[i] = rd.choice(grid)
    return x_star


def compute_delta(num_levels):
    return float(num_levels) / (2 * (num_levels - 1))
