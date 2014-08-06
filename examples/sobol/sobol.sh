#!/bin/bash

# Example: generating samples from the command line
cd ../../ # hack
python -m SALib.sample.saltelli \
	   -n 1000 \
	   -p ./SALib/test_functions/params/Ishigami.txt \
	   -o model_input.txt \
	   --delimiter=' ' \
	   --precision=8 \
     --max-order 2

# Options:
# -p, --paramfile: Your parameter range file (3 columns: parameter name, lower bound, upper bound)
#
# -n, --samples: Sample size. 
#				 Number of model runs is N(2D + 2) if calculating second-order indices (default) 
#        or N(D + 2) otherwise.
#
# -o, --output: File to output your samples into.
# 
# --delimiter (optional): Output file delimiter.
#
# --precision (optional): Digits of precision in the output file. Default is 8.
#
# --max-order (optional): Maximum order of indices to calculate. Choose 1 or 2, default is 2. 
#								   Choosing 1 will reduce total model runs from N(2D + 2) to N(D + 2)
#								   Must use the same value (either 1 or 2) for both sampling and analysis.
