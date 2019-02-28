import sys
sys.path.append('../..')

from SALib.analyze import rbd_fast
from SALib.sample import latin
from SALib.test_functions import Ishigami
from SALib.util import read_param_file

# Read the parameter range file and generate samples
problem = read_param_file('../../src/SALib/test_functions/params/Ishigami.txt')

# Generate samples
param_values = latin.sample(problem, 1000)

# Run the "model" and save the output in a text file
# This will happen offline for external models
Y = Ishigami.evaluate(param_values)

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
Si = rbd_fast.analyze(problem, Y, param_values, print_to_console=False)
# Returns a dictionary with keys 'S1' and 'ST'
# e.g. Si['S1'] contains the first-order index for each parameter, in the
# same order as the parameter file
