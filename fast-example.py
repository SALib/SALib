from SALib import read_param_file
from SALib.sample import fast_sampler, scale_samples
from SALib.test_functions import Sobol_G
from SALib.analyze import extended_fast
import numpy as np
import random as rd

# Example: Run FAST on Sobol G Function

# Set random seed
seed = 1
np.random.seed(seed)
rd.seed(seed)

# Read in parameter range file and generate samples
pf = read_param_file('SGParams.txt')
param_values = fast_sampler.sample(500, pf['num_vars'])
scale_samples(param_values, pf['bounds'])

# Run model and save output (can occur externally)
Y = Sobol_G.evaluate(param_values)
np.savetxt("SGOutput.txt", Y, delimiter=' ')

# Perform FAST analysis on model output
# Specify column of the output file to analyze (zero-indexed)
extended_fast.analyze('SGParams.txt', 'SGOutput.txt', column = 0)
