import sys
sys.path.append('../..')

from SALib.analyze import delta

# Read the parameter range file and generate samples
# Since this is "given data", the bounds in the parameter file will not be used
# but the columns are still expected
param_file = '../../SALib/test_functions/params/Ishigami.txt'

# Perform the sensitivity analysis using the model output
# Specify which column of the output file to analyze (zero-indexed)
Si = delta.analyze(param_file, 'model_input.txt', 'model_output.txt', column = 0, num_resamples=10, conf_level = 0.95, print_to_console=False)
# Returns a dictionary with keys 'delta', 'delta_conf', 'S1', 'S1_conf'
print Si['delta']