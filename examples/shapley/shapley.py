import sys

import numpy as np

from SALib.analyze import shapley
from SALib.sample import shapley as shapley_sampler
from SALib.test_functions import Ishigami
from SALib.util import read_param_file

sys.path.append("../..")


# Goda's estimator assumes independent input variables. It requires
# N * (D + 1) model evaluations for N trajectories and D variables.
problem = read_param_file("../../src/SALib/test_functions/params/Ishigami.txt")
param_values = shapley_sampler.sample(problem, 10_000, seed=100)

# Run the model. External models can evaluate and store these rows offline.
Y = Ishigami.evaluate(param_values)

# Effects are in output-variance units and their sum estimates Var[Y].
Si = shapley.analyze(problem, param_values, Y, print_to_console=True)
normalized_effects = Si["Shapley"] / np.sum(Si["Shapley"])
print("\nNormalized effects:\n", normalized_effects)
