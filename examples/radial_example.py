from SALib.sample.radial import sample as r_sample
from SALib.sample.morris import sample as m_sample
from SALib.sample.latin import sample as l_sample

from SALib.analyze.sobol import analyze as s_analyze
from SALib.analyze.morris import analyze as m_analyze
from SALib.analyze.radial_ee import analyze as r_analyze
from SALib.analyze.radial_st import analyze as rst_analyze

from SALib.test_functions import Ishigami

import numpy as np

problem = {
    'num_vars': 3,
    'names': ['x1', 'x2', 'x3'],
    'bounds': [[-np.pi, np.pi]]*3
}

sample_sets = 1000

r_X = r_sample(problem, sample_sets)
m_X = m_sample(problem, sample_sets)


r_Y = Ishigami.evaluate(r_X)
m_Y = Ishigami.evaluate(m_X)

m_result = m_analyze(problem, m_X, m_Y, num_resamples=10000)
r_result = r_analyze(problem, r_Y, sample_sets, num_resamples=10000)

rst_X = r_sample(problem, 10000)
rst_Y = Ishigami.evaluate(rst_X)
rst_result = rst_analyze(problem, rst_Y, 10000, num_resamples=10000)

l_X = l_sample(problem, 10000)
l_Y = Ishigami.evaluate(l_X)
s_result = s_analyze(problem, l_Y, calc_second_order=False, conf_level=0.95, print_to_console=False)


print(r_result.to_df())
print(m_result.to_df())
print(rst_result.to_df())
print(s_result.to_df()[0])
