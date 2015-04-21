import sys

from SALib.analyze import morris
from SALib.sample.morris import sample
from SALib.test_functions import Ishigami
from SALib.util import read_param_file


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
param_values = sample(problem, N=1000, num_levels=10, grid_jump=5, \
                      optimal_trajectories=None)

# To use optimized trajectories (brute force method), give an integer value for optimal_trajectories

# Run the "model" -- this will happen offline for external models
Y = Ishigami.evaluate(param_values)

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
Si = morris.analyze(problem, param_values, Y, conf_level=0.95, print_to_console=False,
                    num_levels=10, grid_jump=5)
# Returns a dictionary with keys 'mu', 'mu_star', 'sigma', and 'mu_star_conf'
# e.g. Si['mu_star'] contains the mu* value for each parameter, in the
# same order as the parameter file
print(Si['mu_star'])
