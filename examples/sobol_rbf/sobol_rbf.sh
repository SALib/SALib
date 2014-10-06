#!/bin/bash

cd ../../ # hack
# Then use the output to run the analysis. 
# Sensitivity indices will print to command line. Use ">" to write to file.

# Bootstrap CIs are not given here because these only describe the sampling
# uncertainty in the metamodel. With sufficient N_rbf, this is irrelevant.
# The main source of uncertainty is the model fit, given by the cross-validation R^2 value.
# The value 1-R^2 is the variance in Y not accounted for by the metamodel, so 
# this should serve as a sort of error estimate for the sensitivity indices.

python -m SALib.analyze.sobol_rbf \
     -p ./SALib/test_functions/params/Ishigami.txt \
     -X model_input.txt \
     -Y model_output.txt \
     -N 10000 \
     -k 5 \
     -c 0 \

# Options:
# -p, --paramfile: Your parameter range file (3 columns: parameter name, lower bound, upper bound)
#     Parameter bounds should be similar to the min/max of the dataset,
#     otherwise the metamodel will be sampled outside the observation range
#
# -Y, --model-output-file: File of model output values to analyze
#
# -X, --model-input-file: File of model input values (parameter samples).
#
# -N, --N-rbf (optional): Number of samples to perform on the metamodel. Default 100,000.
#
# -k, --n-folds (optional): k for k-fold cross validation. Default 10.
#
# -c, --column (optional): Column of model output file to analyze. 
#                If the file only has one column, this argument will be ignored.
#
# --delimiter (optional): Model output file delimiter.
#
# -t, --training_sample (optional): Number of randomly subsampled observations to use
#     for training the metamodel. Default uses all observations, which can be slow.
#