import sys
sys.path.append('../..')

from SALib.sample import finite_diff
from SALib.analyze import dgsm
from SALib.test_functions import Ishigami
import numpy as np

# Read the parameter range file and generate samples
param_file = '../../SALib/test_functions/params/Ishigami.txt'

# Generate samples
param_values = finite_diff.sample(1000, param_file, delta=0.001)

# Save the parameter values in a file (they are needed in the analysis)
np.savetxt('model_input.txt', param_values, delimiter=' ')

# Run the "model" and save the output in a text file
# This will happen offline for external models
Y = Ishigami.evaluate(param_values)
np.savetxt('model_output.txt', Y, delimiter=' ')

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
Si = dgsm.analyze(param_file, 'model_input.txt', 'model_output.txt',
                  column=0, conf_level=0.95, print_to_console=False)
# Returns a dictionary with keys 'vi', 'vi_std', 'dgsm', and 'dgsm_conf'
# e.g. Si['vi'] contains the sensitivity measure for each parameter, in
# the same order as the parameter file

# For comparison, Morris mu* < sqrt(v_i)
# and total order S_tot <= dgsm, following Sobol and Kucherenko (2009)
