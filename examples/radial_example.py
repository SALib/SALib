from SALib.sample.radial.radial_sobol import sample as r_sample
from SALib.sample.radial.radial_mc import sample as rmc_sample
from SALib.sample.morris import sample as m_sample
from SALib.sample.saltelli import sample as s_sample

from SALib.analyze.sobol import analyze as s_analyze
from SALib.analyze.morris import analyze as m_analyze
from SALib.analyze.radial_ee import analyze as ree_analyze
from SALib.analyze.radial_st import analyze as rst_analyze

from SALib.test_functions import Ishigami

import numpy as np

problem = {
    'num_vars': 3,
    'names': ['x1', 'x2', 'x3'],
    'bounds': [[-np.pi, np.pi]]*3
}

sample_sets = 10000

r_X = r_sample(problem, sample_sets)
rmc_X = rmc_sample(problem, 50000)
m_X = m_sample(problem, sample_sets)


r_Y = Ishigami.evaluate(r_X)
rmc_Y = Ishigami.evaluate(rmc_X)
m_Y = Ishigami.evaluate(m_X)

m_result = m_analyze(problem, m_X, m_Y, num_resamples=1000)

r_result = ree_analyze(problem, r_X, r_Y, sample_sets, num_resamples=1000)
rmc_result = ree_analyze(problem, rmc_X, rmc_Y, 50000, num_resamples=1000)

rst_X = r_sample(problem, 1000)
rst_Y = Ishigami.evaluate(rst_X)
rst_result = rst_analyze(problem, rst_Y, 1000, num_resamples=100)

l_X = s_sample(problem, 1000)
l_Y = Ishigami.evaluate(l_X)
s_result = s_analyze(problem, l_Y, calc_second_order=True, conf_level=0.95, print_to_console=False)


print("EE result for Sobol and MC approach")
print(r_result.to_df())
print(rmc_result.to_df())

print("Morris results")
print(m_result.to_df())

print("Radial ST results")
print(rst_result.to_df())

print("Sobol/Saltelli result")
print(s_result.to_df()[0])
print(s_result.to_df()[1])
