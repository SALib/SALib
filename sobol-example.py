from SALib import read_param_file
from SALib.sample import saltelli, scale_samples
from SALib.test_functions import Sobol_G
from SALib.analyze import sobol
import numpy as np
import random as rd

# Example: Run Sobol SA on a test function (Sobol G Function)
# Second order indices are optional, turned on by default

# Set random seed
seed = 1
np.random.seed(seed)
rd.seed(seed)

# Read the parameter range file and generate samples
param_file = './SALib/test_functions/Sobol_G_Params.txt'
pf = read_param_file(param_file)
param_values = saltelli.sample(10, pf['num_vars'])#, calc_second_order = False)
scale_samples(param_values, pf['bounds'])

# Run the "model" and save the output in a text file
# This will happen offline for external models
Y = Sobol_G.evaluate(param_values)
np.savetxt("SGOutput.txt", Y, delimiter=' ')

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
sobol.analyze(param_file, 'SGOutput.txt', column = 0)#, calc_second_order = False)