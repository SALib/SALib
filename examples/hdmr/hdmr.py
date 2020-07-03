import sys
sys.path.append('../../src')
import numpy as np
from SALib.analyze import hdmr
from SALib.sample import latin
from SALib.test_functions import Ishigami
from SALib.util import read_param_file

# This is the test case taken from Li's paper 
# Genyuan Li et al., Journal of Physical Chemistry A., V. 114 (19), 
# pp. 6022-6032, 2010.

# Read the parameter range file and generate samples
problem = {
  'num_vars': 3,
  'names': ['x1', 'x2', 'x3'],
  'bounds': [[-np.pi, np.pi]]*3
}

# Generate samples
X = latin.sample(problem, 1000, seed=101)

# Run the "model" and save the output in a text file
Y = Ishigami.evaluate(X)

# Add random error
sigma = np.var(Y) / 100

# SALib-HDMR options 
options = {
  'maxorder': 2,
  'maxiter': 100,
  'm': 4,
  'K': 30,
  'R': 500,
  'alpha': 0.95,
  'lambdax': 0.01,
  'print_to_console': True,
  'seed': 101
} 
# Run SALib-HDMR
Si, C = hdmr.analyze(problem, X, Y, **options)

Si.plot()

# Generate samples
X = latin.sample(problem, 5000, seed=int(np.random.random()*100))
# Run the "model" 
Y = Ishigami.evaluate(X)
# Emulator
Y_em = Si.emulate(C, X, Y)