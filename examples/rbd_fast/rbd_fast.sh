#!/bin/bash

# Example: generating samples from the command line
# cd ../../ # hack
# python -m SALib.sample.latin \
# 	   -n 1000 \
# 	   -p ./SALib/test_functions/params/Ishigami.txt \
# 	   -o model_input.txt \
# 	   --delimiter=' ' \
# 	   --precision=8 \
# 		 --seed=100

salib sample latin \
	-n 1000 \
	-p ../../SALib/test_functions/params/Ishigami.txt \
	-o ../data/model_input.txt \
	--delimiter=' ' \
	--precision=8 \
	--seed=100

# Options:
# -p, --paramfile: Your parameter range file (3 columns: parameter name, lower bound, upper bound)
#
# -n, --samples: Sample size.
#				 Number of model runs is N
#
# -o, --output: File to output your samples into.
#
# --delimiter (optional): Output file delimiter.
#
# --precision (optional): Digits of precision in the output file. Default is 8.
#
# -M: RBD-FAST M coefficient, default 4
#
# -s, --seed (optional): Seed value for random number generation

# Run the model using the inputs sampled above, and save outputs
python -c "from SALib.test_functions import Ishigami; import numpy as np; np.savetxt('model_output.txt', Ishigami.evaluate(np.loadtxt('model_input.txt')))"

# Then use the output to run the analysis.
# Sensitivity indices will print to command line. Use ">" to write to file.

# python -m SALib.analyze.rbd_fast \
# 	-p ../../SALib/test_functions/params/Ishigami.txt \
# 	-Y model_output.txt \
# 	-X model_input.txt \
# 	--seed=100

salib analyze rbd_fast \
	-p ../../SALib/test_functions/params/Ishigami.txt \
	-Y ../data/model_output.txt \
	-X ../data/model_input.txt \
	--seed=100

# Options:
# -p, --paramfile: Your parameter range file (3 columns: parameter name, lower bound, upper bound)
#
# -Y, --model-output-file: File of model output values to analyze
# -X, --model-input-file: File of model input values to analyze
#
# --delimiter (optional): Model output file delimiter.
#
# -s, --seed (optional): Seed value for random number generation
