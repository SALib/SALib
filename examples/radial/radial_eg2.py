from SALib.sample.radial.radial_sobol import sample as r_sample
from SALib.sample.radial.radial_mc import sample as rmc_sample

from SALib.sample.morris import sample as m_sample
from SALib.sample.saltelli import sample as s_sample

from SALib.analyze.sobol import analyze as s_analyze
from SALib.analyze.morris import analyze as m_analyze
from SALib.analyze.radial_ee import analyze as ree_analyze

from SALib.analyze.sobol_jansen import analyze as j_analyze

from SALib.test_functions import Sobol_G

import numpy as np

print("G4 - Sobol Jansen Estimator")
problem = {
    'num_vars': 20,
    'names': ['x{}'.format(i) for i in range(1, 21)],
    'bounds': [[0, 1] * 20]
}

sample_sets = 12500

a = np.array([100, 0, 100, 100, 100,
              100, 1, 0, 100, 100, 
              0, 100, 100, 100, 1, 
              100, 100, 0, 100, 1
])

alpha = np.array([1, 4, 1, 1, 1,
                       1, 0.5, 3, 1, 1, 
                       2, 1, 1, 1, 0.5, 
                       1, 1, 1.5, 1, 0.5
])


delta = np.random.rand(20)
r_X = r_sample(problem, sample_sets)
r_Y = Sobol_G.evaluate(r_X, a, alpha=alpha, delta=delta)

g4_result = j_analyze(problem, r_Y, sample_sets, num_resamples=200, seed=101)

ax = g4_result.plot()
ax.axhline(y=0.05)


#### G10 #####

print("G10 - Sobol Jansen Estimator")

a = np.array([100, 0, 100, 100, 100, 
              100, 0, 1, 0, 0, 
              1, 0, 100, 100, 0,
              100, 100, 3, 100, 0])

alpha = np.array([1, 4, 1, 1, 1,
                  1, 0.4, 3, 0.8, 0.7,
                  2, 1.3, 1, 1, 0.5,
                  1, 1, 2.5, 1, 0.6])

# a = np.array([100, 0, 100, 100, 100,
#                   100, 1, 10, 0, 0,
#                   9, 0, 100, 100, 4,
#                   100, 100, 7, 100, 2])

# alpha = np.array([1, 4, 1, 1, 1,
#                     1, 0.4, 3, 0.8, 0.7,
#                     2, 1.3, 1, 1, 0.3,
#                     1, 1, 1.5, 1, 0.6])

delta = np.random.rand(20)

# r_X = r_sample(problem, sample_sets)
r_X = rmc_sample(problem, sample_sets)
r_Y = Sobol_G.evaluate(r_X, a, alpha=alpha, delta=delta)

j_result = j_analyze(problem, r_Y, sample_sets, num_resamples=100)
# j_result = ree_analyze(problem, r_X, r_Y, sample_sets, num_resamples=500)

ax = j_result.plot()
ax.axhline(y=0.05)



## G4 EE analysis

r_result = ree_analyze(problem, r_X, r_Y, sample_sets, num_resamples=1000)
ax = r_result.plot()
ax.axhline(y=0.05)

a = np.array([100, 0, 100, 100, 100,
              100, 1, 0, 100, 100,
              0, 100, 100, 100, 1,
              100, 100, 0, 100, 1
])

alpha_vals = np.array([1, 4, 1, 1, 1,
                       1, 0.5, 3, 1, 1,
                       2, 1, 1, 1, 0.5,
                       1, 1, 1.5, 1, 0.5
])

r_Y = Sobol_G.evaluate(r_X, a, alpha=alpha_vals)
r_result = ree_analyze(problem, r_X, r_Y, sample_sets, num_resamples=1000)
ax = r_result.plot()
ax.axhline(y=0.05)

# G4 again!

j_result = j_analyze(problem, r_Y, sample_sets, num_resamples=200, seed=101)
ax = j_result.plot()
ax.axhline(y=0.05)
