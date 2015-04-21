import sys

from SALib.analyze import dgsm
from SALib.sample import finite_diff
from SALib.test_functions import Ishigami
from SALib.util import read_param_file


sys.path.append('../..')


# Read the parameter range file and generate samples
problem = read_param_file('../../SALib/test_functions/params/Ishigami.txt')

# Generate samples
param_values = finite_diff.sample(problem, 1000, delta=0.001)

# Run the "model" -- this will happen offline for external models
Y = Ishigami.evaluate(param_values)

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
Si = dgsm.analyze(problem, param_values, Y, conf_level=0.95, print_to_console=False)
# Returns a dictionary with keys 'vi', 'vi_std', 'dgsm', and 'dgsm_conf'
# e.g. Si['vi'] contains the sensitivity measure for each parameter, in
# the same order as the parameter file

# For comparison, Morris mu* < sqrt(v_i)
# and total order S_tot <= dgsm, following Sobol and Kucherenko (2009)
