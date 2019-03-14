@echo off

REM Example: generating samples from the command line
salib sample saltelli ^
  -n 1000 ^
  -p ../../src/SALib/test_functions/params/Ishigami.txt ^
  -o ../data/model_input.txt ^
  --delimiter=" " ^
  --precision=8 ^
  --max-order=2 ^
  --seed=100

REM You can also use the module directly through Python
REM python -m SALib.sample.saltelli ^
REM      -n 1000 ^
REM      -p ../../src/SALib/test_functions/params/Ishigami.txt ^
REM      -o ../data/model_input.txt ^
REM      --delimiter=" " ^
REM      --precision=8 ^
REM      --max-order=2 ^
REM      --seed=100

REM Options:
REM -p, --paramfile: Your parameter range file (3 columns: parameter name, lower bound, upper bound)
REM
REM -n, --samples: Sample size.
REM				 Number of model runs is N(2D + 2) if calculating second-order indices (default)
REM        or N(D + 2) otherwise.
REM
REM -o, --output: File to output your samples into.
REM
REM --delimiter (optional): Output file delimiter.
REM
REM --precision (optional): Digits of precision in the output file. Default is 8.
REM
REM --max-order (optional): Maximum order of indices to calculate. Choose 1 or 2, default is 2.
REM								   Choosing 1 will reduce total model runs from N(2D + 2) to N(D + 2)
REM								   Must use the same value (either 1 or 2) for both sampling and analysis.
REM
REM -s, --seed (optional): Seed value for random number generation

REM Run the model using the inputs sampled above, and save outputs
python -c "from SALib.test_functions import Ishigami; import numpy as np; np.savetxt('../data/model_output.txt', Ishigami.evaluate(np.loadtxt('../data/model_input.txt')))"

REM Then use the output to run the analysis.
REM Sensitivity indices will print to command line. Use ">" to write to file.

salib analyze sobol ^
  -p ../../src/SALib/test_functions/params/Ishigami.txt ^
  -Y ../data/model_output.txt ^
  -c 0 ^
  --max-order=2 ^
  -r 1000 ^
  --seed=100

REM python -m SALib.analyze.sobol ^
REM   -p ../../src/SALib/test_functions/params/Ishigami.txt ^
REM   -Y ../data/model_output.txt ^
REM   -c 0 ^
REM   --max-order=2 ^
REM   -r 1000 ^
REM   --seed=100

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
REM --max-order (optional): Maximum order of indices to calculate.
REM               This must match the value chosen during sampling.
REM
REM -r, --resamples (optional): Number of bootstrap resamples used to calculate confidence intervals on indices. Default 1000.
REM
REM
REM -s, --seed (optional): Seed value for random number generation
REM
REM --parallel (optional): Flag to enable parallel execution with multiprocessing
REM
REM --processors (optional, int): Number of processors to be used with the parallel option

REM First-order indices expected with Saltelli sampling:
REM x1: 0.3139
REM x2: 0.4424
REM x3: 0.0