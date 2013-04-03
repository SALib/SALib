# Optional - turn off bytecode (.pyc files)
import sys
sys.dont_write_bytecode = True

from SALib.sample import saltelli, morris_oat, fast_sampler
from SALib.analyze import sobol, morris, extended_fast
from SALib.test_functions import Sobol_G
from SALib.util import scale_samples, read_param_file
import numpy as np
import random as rd

# Example: Run Sobol, Morris, or FAST on a test function (Sobol G Function)
# The parameters shown for each method are also the default values if omitted

# Set random seed
seed = 1
np.random.seed(seed)
rd.seed(seed)

# Read the parameter range file and generate samples
param_file = './SALib/test_functions/Sobol_G_Params.txt'
pf = read_param_file(param_file)

# Generate samples (choose method here)
param_values = saltelli.sample(10, pf['num_vars'], calc_second_order = True)
# param_values = morris_oat.sample(20, pf['num_vars'], num_levels = 10, grid_jump = 5)
# param_values = fast_sampler.sample(500, pf['num_vars'])

# Samples are given in range [0, 1] by default. Rescale them to your parameter bounds.
scale_samples(param_values, pf['bounds'])

# For Method of Morris, save the parameter values in a file (they are needed in the analysis)
# FAST and Sobol do not require this step
# np.savetxt('SGInput.txt', param_values, delimiter=' ')

# Run the "model" and save the output in a text file
# This will happen offline for external models
Y = Sobol_G.evaluate(param_values)
np.savetxt("SGOutput.txt", Y, delimiter=' ')

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
sobol.analyze(param_file, 'SGOutput.txt', column = 0, calc_second_order = True)
# morris.analyze(param_file, 'SGInput.txt', 'SGOutput.txt', column = 0)
# extended_fast.analyze(param_file, 'SGOutput.txt', column = 0)