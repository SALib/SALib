#!/bin/bash

# Example: generating samples from the command line
cd ../../ # hack
python -m SALib.sample.finite_diff \
     -n 1000 \
     -p ./SALib/test_functions/params/Ishigami.txt \
     -r model_input.txt \
     -d 0.001 \
     --delimiter=' ' \
     --precision=8 \
     --seed=100

# Options:
# -p, --paramfile: Your parameter range file (3 columns: parameter name, lower bound, upper bound)
#
# -n, --samples: Sample size.
#				 Number of model runs is N(2D + 2) if calculating second-order indices (default)
#        or N(D + 2) otherwise.
#
# -o, --output: File to output your samples into.
#
# -d, --delta (optional): Finite difference step size (percent). Default is 0.01.
#
# --delimiter (optional): Output file delimiter.
#
# --precision (optional): Digits of precision in the output file. Default is 8.
#
# -s, --seed (optional): Seed value for random number generation

# Run the model using the inputs sampled above, and save outputs
python -c "from SALib.test_functions import Ishigami; import numpy as np; np.savetxt('model_output.txt', Ishigami.evaluate(np.loadtxt('model_input.txt')))"

# Then use the output to run the analysis.
# Sensitivity indices will print to command line. Use ">" to write to file.

python -m SALib.analyze.dgsm \
     -p ./SALib/test_functions/params/Ishigami.txt \
     -X model_input.txt \
     -Y model_output.txt \
     -c 0 \
     -r 1000 \
     --seed=100

# Options:
# -p, --paramfile: Your parameter range file (3 columns: parameter name, lower bound, upper bound)
#
# -Y, --model-output-file: File of model output values to analyze
#
# -X, --model-input-file: File of model input values
#
# -c, --column (optional): Column of model output file to analyze.
#                If the file only has one column, this argument will be ignored.
#
# --delimiter (optional): Model output file delimiter.
#
# -r, --resamples (optional): Number of bootstrap resamples used to calculate confidence intervals on indices. Default 1000.
#
# -s, --seed (optional): Seed value for random number generation
