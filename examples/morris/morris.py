import sys

from SALib.analyze import morris
from SALib.sample.morris import sample
from SALib.test_functions import Sobol_G, Ishigami
from SALib.util import read_param_file
from SALib.plotting.morris import horizontal_bar_plot, covariance_plot, \
                                sample_histograms
import matplotlib.pyplot as plt
import numpy as np


sys.path.append('../..')


# Read the parameter range file and generate samples
problem = read_param_file('../../SALib/test_functions/params/Ishigami.txt')
# or define manually without a parameter file:
# problem = {
#  'num_vars': 3,
#  'names': ['x1', 'x2', 'x3'],
#  'groups': None,
#  'bounds': [[-3.14159265359, 3.14159265359],
#             [-3.14159265359, 3.14159265359],
#             [-3.14159265359, 3.14159265359]]
# }

# Files with a 4th column for "group name" will be detected automatically, e.g.:
# param_file = '../../SALib/test_functions/params/Ishigami_groups.txt'

# Generate samples
param_values = sample(problem, N=10, num_levels=4, grid_jump=2, \
                      optimal_trajectories=None)

# To use optimized trajectories (brute force method), give an integer value for optimal_trajectories

# Run the "model" -- this will happen offline for external models
Y = Ishigami.evaluate(param_values)

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
Si = morris.analyze(problem, param_values, Y, conf_level=0.95, 
                    print_to_console=True,
                    num_levels=4, grid_jump=2, num_resamples=100)
# Returns a dictionary with keys 'mu', 'mu_star', 'sigma', and 'mu_star_conf'
# e.g. Si['mu_star'] contains the mu* value for each parameter, in the
# same order as the parameter file

fig, (ax1, ax2) = plt.subplots(1, 2)
horizontal_bar_plot(ax1, Si,{}, sortby='mu_star', unit=r"tCO$_2$/year")
covariance_plot(ax2, Si, {}, unit=r"tCO$_2$/year")

fig2 = plt.figure()
sample_histograms(fig2, param_values, problem, {'color':'y'})
plt.show()
