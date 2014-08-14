import sys
sys.path.append('../..')

from SALib.analyze import sobol_rbf

# Read the parameter range file and generate samples
# The bounds in the parameter file will be used to sample the metamodel
# generated from model_input -> model_output
param_file = '../../SALib/test_functions/params/Ishigami.txt'

# Build a metamodel with Support Vector Regression (SVR),
# then run Sobol analysis on the metamodel.
# Returns a dictionary with keys 'S1', 'ST', 'S2', 'R2_cv', and 'R2_fullset'
# Where 'R2_cv' is the mean R^2 value from k-fold cross validation,
# 'R2_fullset' is the R^2 value when the metamodel is applied to all observed data,
# and the other entries are lists of size D (the number of parameters)
# containing the indices in the same order as the parameter file

Si = sobol_rbf.analyze(param_file, 'model_input.txt', 'model_output.txt', 
  N_rbf=100000, n_folds = 10, column = 0, print_to_console=False,)

# There is another option called training_sample which specifies the number of subsampled
# observations to use for training the metamodel. By default, all points are used.
# (For larger datasets this will take too long)
