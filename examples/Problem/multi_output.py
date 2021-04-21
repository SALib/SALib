"""Multi-output analysis and plotting example."""

import time

import matplotlib.pyplot as plt
import numpy as np

from SALib.test_functions import lake_problem
from SALib.test_functions import Ishigami
from SALib import ProblemSpec


seed_val = 101

sp = ProblemSpec({
    'names': ['a', 'q', 'b', 'mean', 'stdev', 'delta', 'alpha'],
    'bounds': [[0.0, 0.1], [2.0, 4.5], [0.1, 0.45], [0.01, 0.05], [0.001, 0.005], [0.93, 0.99], [0.2, 0.5]],
    'outputs': ['max_P', 'Utility', 'Inertia', 'Reliability']
})

(sp.sample_saltelli(1000, check_conv=False)
    .evaluate(lake_problem.evaluate)
    .analyze_sobol())

print(sp)

sp.plot()
plt.show()

sp.analyze_delta(num_resamples=5)
sp.plot()
plt.show()


sp = ProblemSpec({
    'names': ['x1', 'x2', 'x3'],
    'bounds': [[-np.pi, np.pi]*3],
    'outputs': ['Y']
})

(sp.sample_saltelli(500, check_conv=False)
    .evaluate(Ishigami.evaluate)
    .analyze_sobol())

sp.plot()
plt.show()