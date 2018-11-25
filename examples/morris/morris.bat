@echo off

REM Example: generating samples from the command line
salib sample morris ^
  -n 100 ^
  -p ../../SALib/test_functions/params/Ishigami.txt ^
  -o ../data/model_input.txt ^
  -lo True ^
  --delimiter=" " ^
  --precision=8 ^
  --levels=10 ^
  --seed=100

REM You can also use the module directly through Python
REM python -m SALib.sample.morris ^
REM        -n 100 ^
REM        -p ../../SALib/test_functions/params/Ishigami.txt ^
REM        -o ../data/model_input.txt ^
REM        -lo True ^
REM        --delimiter=" " ^
REM        --precision=8 ^
REM        --levels=10 ^
REM        --seed=100

REM Options:
REM -p, --paramfile: Your parameter range file
REM                  (3 columns: parameter name,
REM                              lower bound,
REM                              upper bound) with an optional 4th "group" column for Morris only
REM
REM -n, --samples: Sample size.
REM				 Number of model runs is N(D + 1)
REM
REM -o, --output: File to output your samples into.
REM
REM -lo --local: Use local optimization
REM
REM --delimiter (optional): Output file delimiter.
REM
REM --precision (optional): Digits of precision in the output file. Default is 8.
REM
REM -l, --levels (optional): Number of levels in the OAT sampling.
REM                The range of each variable will be discretized into this many levels.
REM                Default is 4.
REM
REM --grid-jump (optional): Grid jump size in the OAT sampling.
REM                         Each variable will be perturbed by this number of levels
REM                         during each trajectory. Default is 2.
REM
REM -k, --k-optimal (optional): Number of optimal trajectories.
REM                             Default behavior uses vanilla OAT if --k-optimal is not specified
REM
REM -s, --seed (optional): Seed value for random number generation

REM Run the model using the inputs sampled above, and save outputs
python -c "from SALib.test_functions import Ishigami; import numpy as np; np.savetxt('../data/model_output.txt', Ishigami.evaluate(np.loadtxt('../data/model_input.txt')))"

REM Then use the output to run the analysis.
REM Sensitivity indices will print to command line. Use ">" to write to file.

salib analyze morris ^
  -p ../../SALib/test_functions/params/Ishigami.txt ^
  -Y ../data/model_output.txt ^
  -c 0 ^
  -X ../data/model_input.txt ^
  -r 1000 ^
  -l=10 ^
  --seed=100

REM python -m SALib.analyze.morris ^
REM   -p ../../SALib/test_functions/params/Ishigami.txt ^
REM   -Y ../data/model_output.txt ^
REM   -c 0 ^
REM   -X ../data/model_input.txt ^
REM   -r 1000 ^
REM   -l=10 ^
REM   --seed=100

REM Options:
REM -p, --paramfile: Your parameter range file
REM                  (3 columns: parameter name,
REM                              lower bound,
REM                              upper bound)
REM
REM -Y, --model-output-file: File of model output values to analyze
REM
REM -c, --column (optional): Column of model output file to analyze.
REM                If the file only has one column, this argument will be ignored.
REM
REM --delimiter (optional): Model output file delimiter.
REM
REM -X, --model-input-file: File of model input values (parameter samples).
REM
REM -r, --resamples (optional): Number of bootstrap resamples used to calculate confidence
REM                             intervals on indices. Default 1000.
REM
REM -s, --seed (optional): Seed value for random number generation
