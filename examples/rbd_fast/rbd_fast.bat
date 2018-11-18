@echo off

REM Example: generating samples from the command line
salib sample latin ^
	-n 1000 ^
	-p ../../SALib/test_functions/params/Ishigami.txt ^
	-r ../data/model_input.txt ^
	--delimiter=" " ^
	--precision=8 ^
	--seed=100

REM You can also use the module directly through Python
REM python -m SALib.sample.latin ^
REM 	   -n 1000 ^
REM 	   -p ./SALib/test_functions/params/Ishigami.txt ^
REM 	   -r model_input.txt ^
REM 	   --delimiter=" " ^
REM 	   --precision=8 ^
REM 		 --seed=100



REM Options:
REM -p, --paramfile: Your parameter range file (3 columns: parameter name, lower bound, upper bound)
REM
REM -n, --samples: Sample size.
REM				 Number of model runs is N
REM
REM -o, --output: File to output your samples into.
REM
REM --delimiter (optional): Output file delimiter.
REM
REM --precision (optional): Digits of precision in the output file. Default is 8.
REM
REM -M: RBD-FAST M coefficient, default 4
REM
REM -s, --seed (optional): Seed value for random number generation

REM Run the model using the inputs sampled above, and save outputs
python -c "from SALib.test_functions import Ishigami; import numpy as np; np.savetxt('../data/model_output.txt', Ishigami.evaluate(np.loadtxt('../data/model_input.txt')))"

REM Then use the output to run the analysis.
REM Sensitivity indices will print to command line. Use ">" to write to file.

salib analyze rbd_fast ^
	-p ../../SALib/test_functions/params/Ishigami.txt ^
	-Y ../data/model_output.txt ^
	-X ../data/model_input.txt ^
	--seed=100

REM python -m SALib.analyze.rbd_fast ^
REM 	-p ../../SALib/test_functions/params/Ishigami.txt ^
REM 	-Y ../data/model_output.txt ^
REM 	-X ../data/model_input.txt ^
REM 	--seed=100

REM Options:
REM -p, --paramfile: Your parameter range file (3 columns: parameter name, lower bound, upper bound)
REM
REM -Y, --model-output-file: File of model output values to analyze
REM -X, --model-input-file: File of model input values to analyze
REM
REM --delimiter (optional): Model output file delimiter.
REM
REM -s, --seed (optional): Seed value for random number generation
