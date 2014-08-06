import sys
sys.path.append('../..')

from SALib.sample import morris_oat
from SALib.analyze import morris
from SALib.test_functions import Ishigami
import numpy as np

# Read the parameter range file and generate samples
param_file = '../../SALib/test_functions/params/Ishigami.txt'

# Generate samples 
param_values = morris_oat.sample(1000, param_file, num_levels = 10, grid_jump = 5)

# Save the parameter values in a file (they are needed in the analysis)
np.savetxt('model_input.txt', param_values, delimiter=' ')

# Run the "model" and save the output in a text file
# This will happen offline for external models
Y = Ishigami.evaluate(param_values)
np.savetxt("model_output.txt", Y, delimiter=' ')

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
Si = morris.analyze(param_file, 'model_input.txt', 'model_output.txt', column = 0, conf_level = 0.95, console=False)
# Returns a dictionary with keys 'mu', 'mu_star', 'sigma', and 'mu_star_conf'
# e.g. Si['mu_star'] contains the mu* value for each parameter, in the same order as the parameter file
