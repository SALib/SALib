import sys
sys.path.append('../..')

from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np

# Read the parameter range file and generate samples
param_file = '../../SALib/test_functions/params/Ishigami.txt'

# Generate samples
param_values = saltelli.sample(1000, param_file, calc_second_order=True)

# Run the "model" and save the output in a text file
# This will happen offline for external models
Y = Ishigami.evaluate(param_values)
np.savetxt('model_output.txt', Y, delimiter=' ')

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
Si = sobol.analyze(param_file, 'model_output.txt', column=0,
                   calc_second_order=True, conf_level=0.95, print_to_console=False)
# Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf'
# e.g. Si['S1'] contains the first-order index for each parameter, in the same order as the parameter file
# The optional second-order indices are now returned in keys 'S2', 'S2_conf'
# These are both upper triangular DxD matrices with nan's in the duplicate
# entries.
