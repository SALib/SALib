from __future__ import division
import numpy as np
import random as rd

#def sample(N, param_file, num_levels, grid_jump):

N = 4
k = 2
num_levels = 4
grid_jump = 2/3

# Create B matrix
B = np.tril(np.ones([k+1, k], dtype=int), -1)
print B

J = np.ones([k+1, k], dtype=int)
print J

delta = num_levels / (2 * (num_levels - 1))
print 'Delta: ', delta

X = np.empty([N*(k+1), k])

x_star = np.empty(k)
for i in range(k):
    x_star[i] = (rd.choice(np.arange(num_levels - grid_jump))) / (num_levels - 1)
x_star = [[1/3.0], [1/3.0]]

# starting point for this trajectory
x_base = np.empty([k+1, k])
for i in range(k):
    x_base[:,i] = (rd.choice(np.arange(num_levels - grid_jump))) / (num_levels - 1)
print 'X_base: \n', x_base


B_hash = J * x_base + delta * B
print 'B_hash: \n', B_hash

D_star = np.diag([rd.choice([-1,1]) for _ in range(k)])
print D_star
D_star[0,0] = 1
D_star[1,1] = -1

# permutation matrix P
perm = np.random.permutation(k)
P = np.zeros([k,k])
for i in range(k):
    P[i, perm[i]] = 1

B_star_1 = (delta/2) * ((2*np.mat(B) - np.mat(J))*np.mat(D_star)+np.mat(J))
B_star_2 = (np.transpose((np.multiply(J[:,0],x_star))) + B_star_1) * np.mat(P)

print 'Test: \n', B_star_1, '\n B*: \n', B_star_2


def test_case():
    k = 2
    p = 4
    delta = 2/3
