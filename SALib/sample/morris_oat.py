from __future__ import division
import numpy as np
import random as rd
from ..util import scale_samples, read_param_file
from . import common_args
    
# Generate N(D + 1) x D matrix of Morris samples (OAT)
def sample(N, param_file, num_levels, grid_jump):
    
    pf = read_param_file(param_file)
    D = pf['num_vars']

    # orientation matrix B: lower triangular (1) + upper triangular (-1)
    B = np.tril(np.ones([D+1, D], dtype=int), -1) + np.triu(-1*np.ones([D+1,D], dtype=int))
    
    # grid step delta, and final sample matrix X
    delta = grid_jump / (num_levels - 1)
    X = np.empty([N*(D+1), D])
    
    # Create N trajectories. Each trajectory contains D+1 parameter sets.
    # (Starts at a base point, and then changes one parameter at a time)
    for j in range(N):
    
        # directions matrix DM - diagonal matrix of either +1 or -1
        DM = np.diag([rd.choice([-1,1]) for _ in range(D)])
    
        # permutation matrix P
        perm = np.random.permutation(D)
        P = np.zeros([D,D])
        for i in range(D):
            P[i, perm[i]] = 1
        
        # starting point for this trajectory
        x_base = np.empty([D+1, D])
        for i in range(D):
            x_base[:,i] = (rd.choice(np.arange(num_levels - grid_jump))) / (num_levels - 1)
        
        # Indices to be assigned to X, corresponding to this trajectory
        index_list = np.arange(D+1) + j*(D + 1)
        delta_diag = np.diag([delta for _ in range(D)])
        
        X[index_list,:] = 0.5*(np.mat(B)*np.mat(P)*np.mat(DM) + 1) * np.mat(delta_diag) + np.mat(x_base)
    
    scale_samples(X, pf['bounds'])
    return X

if __name__ == "__main__":

    parser = common_args.create()
    parser.add_argument('--num-levels', type=int, required=False, default=10, help='Number of grid levels (Morris only)')
    parser.add_argument('--grid-jump', type=int, required=False, default=5, help='Grid jump size (Morris only)')
    args = parser.parse_args()

    np.random.seed(args.seed)
    rd.seed(args.seed)
    param_values = sample(args.samples, args.paramfile, num_levels=args.num_levels, grid_jump=args.grid_jump)
    np.savetxt(args.output, param_values, delimiter=args.delimiter, fmt='%.' + str(args.precision) + 'e')
