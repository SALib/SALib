from SALib import read_param_file
from SALib.sample import morris_oat, scale_samples
from SALib.test_functions import Sobol_G
from SALib.analyze import morris
import numpy as np
import random as rd

# Example: Run Method of Morris on Sobol G Function

# Set random seed
seed = 1
np.random.seed(seed)
rd.seed(seed)

# Read in parameter range file and generate samples
pf = read_param_file('SGParams.txt')
param_values = morris_oat.sample(20, pf['num_vars'], 10, 5)
scale_samples(param_values, pf['bounds'])

# Save the parameter samples (Method of Morris uses them later)
np.savetxt('SGInput.txt', param_values, delimiter=' ')

# Run the model and save output (can occur externally)
Y = Sobol_G.evaluate(param_values)
np.savetxt("SGOutput.txt", Y, delimiter=' ')

# Perform Morris analysis using both model input and output
# Specify column of the output file to analyze (zero-indexed)
morris.analyze('SGParams.txt', 'SGInput.txt', 'SGOutput.txt', column = 0)
