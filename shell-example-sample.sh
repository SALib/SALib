#!/bin/bash

# Generate sensitivity analysis samples from the command line
# To run: ./sample-example.sh
# Note that "python -m" runs the __main__ module in the SALib.sample package

python -m SALib.sample \
	   -m saltelli \
	   -n 100 \
	   -p ./SALib/test_functions/Sobol_G_Params.txt \
	   -o my_output_file.txt \
	   --delimiter=' ' \
	   --precision 8

# Options:
# -p, --paramfile: Your parameter range file (3 columns: parameter name, lower bound, upper bound)
#
# -m, --method: Choose one of {uniform, normal, latin, saltelli, morris, fast}
#				All methods except "normal" assume the ranges are (lower bound, upper bound)
#				"normal" assumes the ranges are (mean, standard deviation)
#
# -n, --samples: Sample size. 
#				 The sample sizes generated for each method are as follows (where D is the number of parameters):
#				 Uniform: N
#				 Normal: N
#				 Latin: N
#				 Saltelli: N(2D + 2) if calculating second-order indices (default) or N(D + 2) otherwise.						
#				 Morris: N(D + 1)
#				 FAST: ND
#
# -o, --output: File to output your samples into
#
# --delimiter (optional): Output file delimiter. Common choices:
#						  Space-delimited (default): --delimiter=' '
#						  Comma-delimited: --delimiter=','
#						  Tab-delimited: --delimiter=$'\t'
#
# --precision (optional): Digits of precision in the output file. Default = 8.
