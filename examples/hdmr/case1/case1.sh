#!/bin/bash

# Sensitivity indices will print to command line. Use ">" to write to file.
salib analyze hdmr \
  -p ../../../src/SALib/test_functions/params/case1.txt \
  -X ../../data/case1_input.txt \
  -Y ../../data/case1_output.txt \
  -c 0 \
  -g 1 \
  -mor 2 \
  -mit 100 \
  -m 2 \
  -K 20 \
  -R 500 \
  -a 0.95 \
  -lambda 0.05 \
  -print 1

# Then use the output to run the analysis.

# You can also use the module directly through Python
# python -m SALib.analyze.hdmr \
#      -p ../../../src/SALib/test_functions/params/case1.txt \
#      -X ../../data/case1_input.txt \
#      -Y ../../data/case1_output.txt \
#      -c 0 \
#      -g 1 \
#      -mor 2 \
#      -mit 100 \
#      -m 2 \
#      -K 20 \
#      -R 500 \
#      -a 0.95 \
#      -lambda 0.05 \
#      -print 1

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
# -g, --graphics (optional): Whether to print out graphics or not (1,0). Default is 1.
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
# -a, --alfa (optional): Confidence interval. Default is 0.95
#
# -lambda, --lambdax (optional): Regularization constant. Default is 0.01
#
# -print, --print-to-console (optional): Whether to print out results to the screen or not (1,0). Default is 1.
