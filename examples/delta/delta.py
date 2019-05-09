import sys

from SALib.analyze import delta
from SALib.util import read_param_file
import numpy as np


sys.path.append('../..')


# Read the parameter range file and generate samples
# Since this is "given data", the bounds in the parameter file will not be used
# but the columns are still expected
problem = read_param_file('../../src/SALib/test_functions/params/Ishigami.txt')
X = np.loadtxt('model_input.txt')
Y = np.loadtxt('model_output.txt')

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
Si = delta.analyze(problem, X, Y, num_resamples=10, conf_level=0.95, print_to_console=False)
# Returns a dictionary with keys 'delta', 'delta_conf', 'S1', 'S1_conf'
print(str(Si['delta']))
