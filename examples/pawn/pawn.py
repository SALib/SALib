import sys

sys.path.append("../..")

from SALib.analyze import pawn
from SALib.sample import latin
from SALib.test_functions import Ishigami
from SALib.util import read_param_file

# Read the parameter range file and generate samples
problem = read_param_file("../../src/SALib/test_functions/params/Ishigami.txt")

# Generate samples
param_values = latin.sample(problem, 1000)

# Run the "model" and save the output in a text file
# This will happen offline for external models
Y = Ishigami.evaluate(param_values)

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
Si = pawn.analyze(problem, param_values, Y, S=10, print_to_console=True)
# Returns a dictionary with key 'PAWN'
# e.g. Si['PAWN'] contains the total-order index for each parameter, in the
# same order as the parameter file
