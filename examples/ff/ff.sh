#!/bin/bash

# Example: generating samples from the command line
salib sample ff \
  -p ../../src/SALib/test_functions/params/Ishigami.txt \
  -o ../data/model_input.txt \
  -n 100 \
  --delimiter=' ' \
  --precision=8 \
  --seed=100

# You can also use the module directly through Python
# python -m SALib.sample.ff \
#        -p ../../src/SALib/test_functions/params/Ishigami.txt \
#        -o ../data/model_input.txt \
#        -n 100 \
#        --delimiter=' ' \
#        --precision=8 \
#        --seed=100

# Options:
# -p, --paramfile: Your parameter range file
#                  (3 columns: parameter name,
#                              lower bound,
#                              upper bound) with an optional 4th "group" column for Morris only
#
# -o, --output: File to output your samples into.
#
# --delimiter (optional): Output file delimiter.
#
# --precision (optional): Digits of precision in the output file. Default is 8.
#
# -s, --seed (optional): Seed value for random number generation

# Run the model using the inputs sampled above, and save outputs
python -c "from SALib.test_functions import Ishigami; import numpy as np; np.savetxt('../data/model_output.txt', Ishigami.evaluate(np.loadtxt('../data/model_input.txt')))"

# Then use the output to run the analysis.
# Sensitivity indices will print to command line. Use ">" to write to file.

salib analyze ff \
  -p ../../src/SALib/test_functions/params/Ishigami.txt \
  -Y ../data/model_output.txt \
  -c 0 \
  -X ../data/model_input.txt \
  --seed=100

# python -m SALib.analyze.ff \
#   -p ../../src/SALib/test_functions/params/Ishigami.txt \
#   -Y ../data/model_output.txt \
#   -c 0 \
#   -X ../data/model_input.txt \
#   --seed=100

# Options:
# -p, --paramfile: Your parameter range file
#                  (3 columns: parameter name,
#                              lower bound,
#                              upper bound)
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
# -s, --seed (optional): Seed value for random number generation
