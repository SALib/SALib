import sys
sys.path.append('../..')

from SALib.sample.morris import sample
from SALib.analyze import morris
from SALib.test_functions import Ishigami
from SALib.util import read_param_file
import numpy as np

# Read the parameter range file and generate samples
param_file = '../../SALib/test_functions/params/Ishigami.txt'
problem = read_param_file(param_file)
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
param_values = sample(problem, N=10000, num_levels=10, grid_jump=5, \
                      optimal_trajectories=None)

# To use optimized trajectories (brute force method), give an integer value for optimal_trajectories

# Save the parameter values in a file (they are needed in the analysis)
np.savetxt('model_input.txt', param_values, delimiter=' ')

# Run the "model" and save the output in a text file
# This will happen offline for external models
Y = Ishigami.evaluate(param_values)
np.savetxt("model_output.txt", Y, delimiter=' ')

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
Si = morris.analyze(param_file, 'model_input.txt', 'model_output.txt',
                    column=0, conf_level=0.95, print_to_console=True,
                    num_levels=10, grid_jump=5)
# Returns a dictionary with keys 'mu', 'mu_star', 'sigma', and 'mu_star_conf'
# e.g. Si['mu_star'] contains the mu* value for each parameter, in the
# same order as the parameter file
print(Si['mu_star'])
