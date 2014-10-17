import cProfile, pstats, StringIO
from esme import find_optimum_trajectories, morris_sample, read_param_file

pr = cProfile.Profile()
pr.enable()

param_file = 'esme_param.txt'
pf = read_param_file(param_file)
num_params = pf['num_vars']
bounds = pf['bounds']

N = 100
k_choices = 4
p_levels = 4
grid_step = 2
# Generates N(D + 1) x D matrix of samples
input_sample = morris_sample(N, num_params, bounds, p_levels, grid_step)
optimal_model_input = find_optimum_trajectories(input_sample, N, num_params, k_choices)

pr.disable()
s = StringIO.StringIO()
sortby = 'cumtime'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print "{} from {}".format(k_choices, N)
print s.getvalue()
