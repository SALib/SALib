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
salib analyze enhanced_hdmr \
  -p ../../src/SALib/test_functions/params/Ishigami.txt \
  -X ../data/model_input.txt \
  -Y ../data/model_output.txt \
  -c 0 \
  -mor 2 \
  -por 7 \
  -K 20 \
  -R 500 \
  -mit 100 \
  -l2 0.01 \
  -a 0.95 \
  -ext True \
  -print True \
  -emul True

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
# -mor, --max-order (optional): Maximum order of HDMR expansion (1,2,3). Default is 2.
#
# -por, --poly-order (optional): Maximum polynomial order (1-10). Default is 3.
#
# -K, --bootstrap (optional): Number of bootstrap (1-100). Default is 20.
#
# -R, --subset (optional): Subsample size (300-N). Default is N/2.
#
# -mit, --max-iter (optional): Maximum iteration number for backfitting algorithm (1-1000). Default is 100.
#
# -l2, --l2-penalty (optional): Regularization term. Default is 0.01.
#
# -a, --alpha (optional): Confidence interval for F-Test. Default is 0.95.
#
# -ext, --extended-base (optional): Whether to use extended base matrix. Default is True.
#
# -print, --print-to-console (optional): Whether to print out result to the console. Default is False
#
# -emul, --return-emulator (optional): Whether to attach emulate() method to the ResultDic. Default is False
