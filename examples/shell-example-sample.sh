#!/bin/bash

# Generate sensitivity analysis samples from the command line
# To run: ./shell-example-sample.sh
# Note that "python -m" runs the __main__ module in the SALib.sample package
# The -B flag prevents the interpreter from compiling bytecode (.pyc files)

python -B -m SALib.sample \
	   -m saltelli \
	   -n 100 \
	   -p ./SALib/test_functions/params/Sobol_G.txt \
	   -o my_samples_file.txt \
	   --delimiter=' ' \
	   --precision=8 \

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
#				 FAST: ND ; must choose N > 4M^2 (N > 64 under default settings)
#
# -o, --output: File to output your samples into.
#
# -s, --seed: Random seed (optional). Will only affect results for Method of Morris.
# 
# --delimiter (optional): Output file delimiter. Common choices:
#						  Space-delimited (default): --delimiter=' '
#						  Comma-delimited: --delimiter=','
#						  Tab-delimited: --delimiter=$'\t'
#
# --precision (optional): Digits of precision in the output file. Default is 8.
#
# --saltelli-max-order (optional): Maximum order of indices to calculate. Only applies for -m saltelli. Choose 1 or 2, default is 2. 
#								   Choosing 1 will reduce total model runs from N(2D + 2) to N(D + 2)
#								   Must use the same value (either 1 or 2) for both sampling and analysis.
#
# --morris-num-levels (optional): Number of levels in the OAT sampling. The range of each variable will be discretized into this many levels. Only applies for -m morris. Default is 10.
#
# --morris-grid-jump (optional): Grid jump size in the OAT sampling. Each variable will be perturbed by this number of levels during each trajectory. Only applies for -m morris. Default is 5.
