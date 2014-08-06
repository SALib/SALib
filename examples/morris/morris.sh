#!/bin/bash

# Example: generating samples from the command line
cd ../../ # hack
python -m SALib.sample.morris_oat \
	   -n 1000 \
	   -p ./SALib/test_functions/params/Ishigami.txt \
	   -o model_input.txt \
	   --delimiter=' ' \
	   --precision=8 \
     --num-levels=10 \
     --grid-jump=5

# Options:
# -p, --paramfile: Your parameter range file (3 columns: parameter name, lower bound, upper bound)
#
# -n, --samples: Sample size. 
#				 Number of model runs is N(D + 1)
#
# -o, --output: File to output your samples into.
# 
# --delimiter (optional): Output file delimiter.
#
# --precision (optional): Digits of precision in the output file. Default is 8.
#
# --morris-num-levels (optional): Number of levels in the OAT sampling. The range of each variable will be discretized into this many levels. Default is 10.
#
# --morris-grid-jump (optional): Grid jump size in the OAT sampling. Each variable will be perturbed by this number of levels during each trajectory. Default is 5.

# Run the model using the inputs sampled above, and save outputs
python -c "from SALib.test_functions import Ishigami; import numpy as np; np.savetxt('model_output.txt', Ishigami.evaluate(np.loadtxt('model_input.txt')))"

# Then use the output to run the analysis. 
# Sensitivity indices will print to command line. Use ">" to write to file.

python -m SALib.analyze.morris \
     -p ./SALib/test_functions/params/Ishigami.txt \
     -Y model_output.txt \
     -c 0 \
     -X model_input.txt \
     -r 1000

# Options:
# -p, --paramfile: Your parameter range file (3 columns: parameter name, lower bound, upper bound)
#
# -Y, --model-output-file: File of model output values to analyze
#
# -c, --column (optional): Column of model output file to analyze. 
#                If the file only has one column, this argument will be ignored.
#
# --delimiter (optional): Model output file delimiter.
#
# -X, --model-input-file: File of model input values (parameter samples).
#
# -r, --resamples (optional): Number of bootstrap resamples used to calculate confidence intervals on indices. Default 1000.
