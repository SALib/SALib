import cProfile, pstats, StringIO
from esme import *
from SALib.sample import morris_oat

pr = cProfile.Profile()
pr.enable()



param_file = 'esme_param.txt'
pf = read_param_file(param_file)
N = 6
num_params = pf['num_vars']
k_choices = 2
p_levels = 4
grid_step = 2
# Generates N(D + 1) x D matrix of samples
input_data = morris_oat.sample(N,
                                param_file,
                                num_levels=p_levels,
                                grid_jump=grid_step)
scores, combos = find_most_distant(input_data, N, num_params, k_choices)
print find_maximum(scores, combos)



pr.disable()
s = StringIO.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print s.getvalue()
