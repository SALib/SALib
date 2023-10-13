import sys

sys.path.append("../..")

from SALib.analyze import fast
from SALib.sample import fast_sampler
from SALib.test_functions import Ishigami
from SALib.util import read_param_file

# Read the parameter range file and generate samples
problem = read_param_file("../../src/SALib/test_functions/params/Ishigami.txt")

# Generate samples
param_values = fast_sampler.sample(problem, 1000, seed=100)

# Run the "model" and save the output in a text file
# This will happen offline for external models
Y = Ishigami.evaluate(param_values)

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
Si = fast.analyze(problem, Y, print_to_console=True, seed=100)
# Returns a dictionary with keys 'S1' and 'ST'
# e.g. Si['S1'] contains the first-order index for each parameter, in the
# same order as the parameter file
