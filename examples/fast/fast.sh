#!/bin/bash

# Example: generating samples from the command line
cd ../../ # hack
python -m SALib.sample.fast_sampler \
	   -n 1000 \
	   -p ./SALib/test_functions/params/Ishigami.txt \
	   -o model_input.txt \
	   --delimiter=' ' \
	   --precision=8 \
     -M 4

# Options:
# -p, --paramfile: Your parameter range file (3 columns: parameter name, lower bound, upper bound)
#
# -n, --samples: Sample size. 
#				 Number of model runs is ND ; must choose N > 4M^2 (N > 64 under default settings)
#
# -o, --output: File to output your samples into.
# 
# --delimiter (optional): Output file delimiter.
#
# --precision (optional): Digits of precision in the output file. Default is 8.
#
# -M: FAST M coefficient, default 4
