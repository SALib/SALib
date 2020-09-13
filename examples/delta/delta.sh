#!/bin/bash

# Sensitivity indices will print to command line. Use ">" to write to file.
salib analyze delta \
  -p ../../src/SALib/test_functions/params/Ishigami.txt \
  -X ../data/model_input.txt \
  -Y ../data/model_output.txt \
  -c 0 \
  -r 10 \
  --seed=100

# Then use the output to run the analysis.

# You can also use the module directly through Python
# python -m SALib.analyze.delta \
#      -p ../../src/SALib/test_functions/params/Ishigami.txt \
#      -X ../data/model_input.txt \
#      -Y ../data/model_output.txt \
#      -c 0 \
#      -r 10 \
#      --seed=100

# Options:
# -p, --paramfile: Your parameter range file (3 columns: parameter name, lower bound, upper bound)
#
# -Y, --model-output-file: File of model output values to analyze
#
# -X, --model-input-file: File of model input values (parameter samples).
#
# -c, --column (optional): Column of model output file to analyze.
#                If the file only has one column, this argument will be ignored.
#
# --delimiter (optional): Model output file delimiter.
#
# -r, --resamples (optional): Number of bootstrap resamples used to calculate confidence intervals on indices. Default 1000.
#
# -s, --seed (optional): Seed value for random number generation
