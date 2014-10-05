import cProfile, pstats, StringIO
from esme import *

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


pr = cProfile.Profile()
pr.enable()

param_file = 'esme_param.txt'
pf = read_param_file(param_file)
N = 30
num_params = pf['num_vars']
bounds = pf['bounds']
k_choices = 5
p_levels = 4
grid_step = 2
# Generates N(D + 1) x D matrix of samples
input_sample = morris_sample(N, num_params, bounds, p_levels, grid_step)

optimal_model_input = find_optimimum_trajectories(input_sample, N, num_params, k_choices)

pr.disable()
s = StringIO.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print s.getvalue()
