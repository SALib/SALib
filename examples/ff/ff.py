import sys

from SALib.analyze.ff import analyze
from SALib.sample.ff import sample
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

# Generate samples
X = sample(problem)

# Run the "model" -- this will happen offline for external models
Y = X[:, 0] + (0.1 * X[:, 1]) + ((1.2 * X[:, 2]) * (0.2 + X[:, 0]))

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
analyze(problem, X, Y, second_order=True, print_to_console=True)
# Returns a dictionary with keys 'ME' (main effect) and 'IE' (interaction effect)
# The techniques bulks out the number of parameters with dummy parameters to the
# nearest 2**n.  Any results involving dummy parameters should be treated with
# a sceptical eye.