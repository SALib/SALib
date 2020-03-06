from SALib.sample.radial import radial_sobol, radial_mc
from SALib.sample import saltelli
from SALib.sample.morris import sample as m_sample
from SALib.analyze import sobol, morris, radial_ee, sobol_jansen

from SALib.test_functions import Ishigami
from SALib.util import read_param_file

problem = read_param_file('../../src/SALib/test_functions/params/Ishigami.txt')

sample_sets = 5000
mc_sample_sets = 5000
sobol_sample = 1000

# Radial Sobol
radial_sobol_param = radial_sobol.sample(problem, sample_sets)
radial_sobol_Y = Ishigami.evaluate(radial_sobol_param)
radial_sobol_result = radial_ee.analyze(problem, radial_sobol_param, radial_sobol_Y, sample_sets)

# Radial MC 
radial_mc_param = radial_mc.sample(problem, mc_sample_sets)
radial_mc_Y = Ishigami.evaluate(radial_mc_param)
radial_mc_result = radial_ee.analyze(problem, radial_mc_param, radial_mc_Y, mc_sample_sets)

# Sobol Jansen
sobol_jansen_result = sobol_jansen.analyze(problem, radial_sobol_Y, sample_sets)

# Morris
morris_param = m_sample(problem, sample_sets)
morris_Y = Ishigami.evaluate(morris_param)
morris_result = morris.analyze(problem, morris_param, morris_Y)

# Sobol
sobol_param = saltelli.sample(problem, sobol_sample)
sobol_Y = Ishigami.evaluate(sobol_param)
Si = sobol.analyze(problem, sobol_Y, calc_second_order=True, conf_level=0.95, print_to_console=False)


print("EE results for Radial Sobol")
print(radial_sobol_result.to_df())

print("EE results for Radial MC")
print(radial_mc_result.to_df())

print("Jansen results for Radial Sobol")
print(sobol_jansen_result.to_df())

print("Morris results")
print(morris_result.to_df())

print("Sobol/Saltelli result")
total_si, first_si, second_si = Si.to_df()
print(total_si)
print(first_si)

