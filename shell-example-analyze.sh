#!/bin/bash

# Perform sensitivity analysis from the command line
# To run: ./shell-example-analyze.sh
# Note that "python -m" runs the __main__ module in the SALib.analyze package
# The -B flag prevents the interpreter from compiling bytecode (.pyc files)

python -B -m SALib.analyze \
	   -m sobol \
	   -p ./SALib/test_functions/Sobol_G_Params.txt \
	   -Y SGOutput.txt \
	   -c 0 \

# By default, this command will print to the terminal.
# To save to a file, use the ">" operator.
	   
# Options:
# -p, --paramfile: Your parameter range file (3 columns: parameter name, lower bound, upper bound)
#
# -m, --method: Choose one of {uniform, normal, latin, saltelli, morris, fast}
#				All methods except "normal" assume the ranges are (lower bound, upper bound)
#				"normal" assumes the ranges are (mean, standard deviation)
#
# -Y, --model-output-file: File of model output values to analyze
#
# -c, --column (optional): Column of model output file to analyze. 
# 						   If the file only has one column, this argument will be ignored.
#
# --delimiter (optional): Model output file delimiter. Common choices:
#						  Space-delimited (default): --delimiter=' '
#						  Comma-delimited: --delimiter=','
#						  Tab-delimited: --delimiter=$'\t'
#
# --sobol-max-order (optional): Maximum order of indices to calculate. Only applies for -m sobol. Choose 1 or 2, default is 2. 
#								This must match the value chosen during sampling.
#
# -X, --morris-model-input (required for Method of Morris only): File of model input values (parameter samples).
#
# -r, --sobol-bootstrap-resamples (optional): Number of bootstrap resamples used to calculate confidence intervals on Sobol indices.