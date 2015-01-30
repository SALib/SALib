import sys
sys.path.append('../..')

from SALib.analyze import sobol_rbf
from SALib.util import read_param_file
import numpy as np

# Read the parameter range file and generate samples
# The bounds in the parameter file will be used to sample the metamodel
# generated from model_input -> model_output
problem = read_param_file('../../SALib/test_functions/params/Ishigami.txt')
X = np.loadtxt('model_input.txt')
Y = np.loadtxt('model_output.txt')

# Build a metamodel with Support Vector Regression (SVR),
# then run Sobol analysis on the metamodel.
# Returns a dictionary with keys 'S1', 'ST', 'S2', 'R2_cv', and 'R2_fullset'
# Where 'R2_cv' is the mean R^2 value from k-fold cross validation,
# 'R2_fullset' is the R^2 value when the metamodel is applied to all observed data,
# and the other entries are lists of size D (the number of parameters)
# containing the indices in the same order as the parameter file

Si = sobol_rbf.analyze(problem, X, Y,
                       N_rbf=100000, n_folds=2, print_to_console=True, training_sample=100)

# There is another option called training_sample which specifies the number of subsampled
# observations to use for training the metamodel. By default, all points are used.
# (For larger datasets this will take too long)
