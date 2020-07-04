#!/bin/bash

# Example: generating samples from the command line
salib sample latin \
	-n 1000 \
	-p ../../src/SALib/test_functions/params/Ishigami.txt \
	-o ../data/model_input.txt \
	--delimiter=" " \
	--precision=8 \
	--seed=100

# Run model and save output
python -c "from SALib.test_functions import Ishigami; import numpy as np; np.savetxt('../data/model_output.txt', Ishigami.evaluate(np.loadtxt('../data/model_input.txt')))"

# Perform hdmr analysis
salib analyze hdmr \
  -p ../../src/SALib/test_functions/params/Ishigami.txt \
  -X ../data/model_input.txt \
  -Y ../data/model_output.txt \
  -c 0 \
  -mor 2 \
  -mit 100 \
  -m 4 \
  -K 20 \
  -R 500 \
  -a 0.95 \
  -lambda 0.01

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
# -mor, --maxorder (optional): Maximum order of HDMR expansion (1,2,3). Default is 2.
#
# -mit, --maxiter (optional): Maximum iteration number for backfitting algorithm (1-1000). Default is 100.
#
# -m, --m-int (optional): Number of B-spline intervals. Default is 2.
#
# -K, --K-bootstrap (optional): Number of bootstrap (1,2,3). Default is 20.
#
# -R, --R-subsample (optional): Subsample size. Default is N/2.
#
# -a, --alpha (optional): Confidence interval. Default is 0.95
#
# -lambda, --lambdax (optional): Regularization constant. Default is 0.01
