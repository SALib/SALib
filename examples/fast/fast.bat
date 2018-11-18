@echo off

REM Example: generating samples from the command line
salib sample fast_sampler ^
	-n 1000 ^
	-p ../../SALib/test_functions/params/Ishigami.txt ^
	-r ../data/model_input.txt ^
	--delimiter=" " ^
	--precision=8 ^
	-M 4 ^
	--seed=100

REM You can also use the module directly through Python
REM python -m SALib.sample.fast_sampler ^
REM 	   -n 1000 ^
REM 	   -p ../../SALib/test_functions/params/Ishigami.txt ^
REM 	   -r ../data/model_input.txt ^
REM 	   --delimiter=' ' ^
REM 	   --precision=8 ^
REM      -M 4 ^
REM 		 --seed=100

REM Options:
REM -p, --paramfile: Your parameter range file (3 columns: parameter name, lower bound, upper bound)
REM
REM -n, --samples: Sample size.
REM				 Number of model runs is ND ; must choose N > 4M^2 (N > 64 under default settings)
REM
REM -o, --output: File to output your samples into.
REM
REM --delimiter (optional): Output file delimiter.
REM
REM --precision (optional): Digits of precision in the output file. Default is 8.
REM
REM -M: FAST M coefficient, default 4
REM
REM -s, --seed (optional): Seed value for random number generation

REM Run the model using the inputs sampled above, and save outputs
python -c "from SALib.test_functions import Ishigami; import numpy as np; np.savetxt('../data/model_output.txt', Ishigami.evaluate(np.loadtxt('../data/model_input.txt')))"

REM Then use the output to run the analysis.
REM Sensitivity indices will print to command line. Use ">" to write to file.

salib analyze fast ^
	-p ../../SALib/test_functions/params/Ishigami.txt ^
	-Y ../data/model_output.txt ^
	-c 0 ^
	--seed=100

REM python -m SALib.analyze.fast ^
REM 	-p ../../SALib/test_functions/params/Ishigami.txt ^
REM 	-Y ../data/model_output.txt ^
REM 	-c 0 ^
REM 	--seed=100

REM Options:
REM -p, --paramfile: Your parameter range file (3 columns: parameter name, lower bound, upper bound)
REM
REM -Y, --model-output-file: File of model output values to analyze
REM
REM -c, --column (optional): Column of model output file to analyze.
REM                If the file only has one column, this argument will be ignored.
REM
REM --delimiter (optional): Model output file delimiter.
REM
REM -s, --seed (optional): Seed value for random number generation
