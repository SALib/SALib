# Optional - turn off bytecode (.pyc files)
import sys
sys.dont_write_bytecode = True

from SALib.sample import saltelli, morris_oat, fast_sampler
from SALib.analyze import sobol, morris, extended_fast
from SALib.test_functions import Sobol_G
from SALib.util import scale_samples, read_param_file
import numpy as np, matplotlib.pyplot as plt
import random as rd

# Example: Run Sobol, Morris, or FAST on a test function (Sobol G Function)
# The settings shown for each method are also the default values if omitted

# Set random seed (does not affect quasi-random Sobol sampling)
seed = 1
np.random.seed(seed)
rd.seed(seed)

# Read the parameter range file and generate samples
param_file = './SALib/test_functions/params/Sobol_G.txt'
pf = read_param_file(param_file)

# Generate samples (choose method here)
# param_values = saltelli.sample(100, pf['num_vars'], calc_second_order = True)
# param_values = morris_oat.sample(100, pf['num_vars'], num_levels = 10, grid_jump = 5)
param_values = fast_sampler.sample(100, pf['num_vars'])

# Samples are given in range [0, 1] by default. Rescale them to your parameter bounds. (If using normal distributions, use "scale_samples_normal" instead)
scale_samples(param_values, pf['bounds'])

# For Method of Morris, save the parameter values in a file (they are needed in the analysis)
# FAST and Sobol do not require this step, unless you want to save samples to input into an external model
# np.savetxt('SGInput.txt', param_values, delimiter=' ')

# Run the "model" and save the output in a text file
# This will happen offline for external models
Y = Sobol_G.evaluate(param_values)
np.savetxt("SGOutput.txt", Y, delimiter=' ')

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
# Si = sobol.analyze(param_file, 'SGOutput.txt', column = 0, calc_second_order = True)
# Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf'
# The optional second-order indices are not returned at this time; refer to the terminal output

# Si = morris.analyze(param_file, 'SGInput.txt', 'SGOutput.txt', column = 0, conf_level = 0.95)
# Returns a dictionary with keys 'mu', 'mu_star', 'sigma', and 'mu_star_conf'

Si = extended_fast.analyze(param_file, 'SGOutput.txt', column = 0)
# Returns a dictionary with keys 'S1' and 'ST'

#Plot the indices. Morris method only (for now)

plt.figure()
plt.plot(Si['S1'], Si['ST'], 'ok')
plt.xlabel('S1')
plt.ylabel('ST')

for i in xrange(len(pf['names'])):
   plt.text(Si['S1'][i]+0.04, Si['ST'][i]+0.02, pf['names'][i])

plt.show()