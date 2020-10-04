import sys
sys.path.append('../..')

from SALib.analyze import pawn
from SALib.sample import latin
from SALib.test_functions import Ishigami
from SALib.util import read_param_file

import numpy as np

# Read the parameter range file and generate samples
problem = read_param_file('../../src/SALib/test_functions/params/Ishigami.txt')

# Generate samples
X = latin.sample(problem, 500)

# Run the "model" and save the output in a text file
# This will happen offline for external models
Y = Ishigami.evaluate(X, A=2.0, B=1.0)

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
Si = pawn.analyze(problem, X, Y, S=10, stat='median', 
                  num_resamples=100, print_to_console=True)
# Returns a dictionary with key 'PAWNi' and 'PAWNi_conf'
# e.g. Si['PAWNi'] contains the PAWN index for each parameter, in the
# same order as the parameter file

# To display plots:
# import matplotlib.pyplot as plt
# Si.plot()
# plt.show()

# Expected sensitivities, according to Pianosi and Wagener (2018)
# with 500 samples
# https://doi.org/10.1016/j.envsoft.2018.07.019
# x1: 0.50
# x2: 0.16
# x3: 0.29
