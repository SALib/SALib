import numpy as np
from SALib.analyze import hdmr
from SALib.sample import latin
from SALib.test_functions import linear_model_1
from SALib.util import read_param_file

# This is the test case from Li's paper 
# Genyuan Li et al., Journal of Physical Chemistry A., V. 114 (19), 
# pp. 6022-6032, 2010.

# Read the parameter range file and generate samples
problem = {'num_vars': 5,'names': ['x1', 'x2', 'x3', 'x4', 'x5'],'bounds': [[0, 1]]*5}
# Generate correlated samples
mean = np.zeros((problem['num_vars']))
cov = [[1,0.6,0.2,0,0],[0.6,1,0.2,0,0],[0.2,0.2,1,0,0],[0,0,0,1,0.2],[0,0,0,0.2,1]]
X = np.random.multivariate_normal(mean, cov, 1000)
# Run the "model" and save the output in a text file
Y = linear_model_1.evaluate(X)
# SALib-HDMR options 
options = {'graphics': 0,'maxorder': 2,'maxiter': 100,'m': 2,'K': 20,'R': 1000,'alfa': 0.95,'lambdax': 0.01,'print_to_console': 1} 
# Run SALib-HDMR
Si = hdmr.analyze(problem,X,Y,options)