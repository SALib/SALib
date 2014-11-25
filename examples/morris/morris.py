import sys
sys.path.append('../..')

from SALib.sample.morris import Morris
from SALib.analyze import morris
from SALib.test_functions import Ishigami
import numpy as np

# Read the parameter range file and generate samples
param_file = '../../SALib/test_functions/params/Ishigami.txt'

# Generate samples
param_values = Morris(param_file, samples=1000, num_levels=10, grid_jump=5, \
                      group_file=None, \
                      optimal_trajectories=None)

# Save the parameter values in a file (they are needed in the analysis)
param_values.save_data('model_input.txt')

# Run the "model" and save the output in a text file
# This will happen offline for external models

online_model_values = param_values.get_input_sample_scaled()
Y = Ishigami.evaluate(online_model_values)
np.savetxt("model_output.txt", Y, delimiter=' ')

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
Si = morris.analyze(param_file, 'model_input.txt', 'model_output.txt',
                    column=0, conf_level=0.95, print_to_console=False)
# Returns a dictionary with keys 'mu', 'mu_star', 'sigma', and 'mu_star_conf'
# e.g. Si['mu_star'] contains the mu* value for each parameter, in the
# same order as the parameter file
print(Si['mu_star'])
