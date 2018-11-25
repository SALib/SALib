@echo off

REM Example: generating samples from the command line
salib sample ff ^
  -p ../../SALib/test_functions/params/Ishigami.txt ^
  -o ../data/model_input.txt ^
  -n 100 ^
  --delimiter=" " ^
  --precision=8 ^
  --seed=100

REM You can also use the module directly through Python
REM python -m SALib.sample.ff ^
REM        -p ../../SALib/test_functions/params/Ishigami.txt ^
REM        -o ../data/model_input.txt ^
REM        -n 100 ^
REM        --delimiter=" " ^
REM        --precision=8 ^
REM        --seed=100

REM Options:
REM -p, --paramfile: Your parameter range file
REM                  (3 columns: parameter name,
REM                              lower bound,
REM                              upper bound) with an optional 4th "group" column for Morris only
REM
REM -o, --output: File to output your samples into.
REM
REM --delimiter (optional): Output file delimiter.
REM
REM --precision (optional): Digits of precision in the output file. Default is 8.
REM
REM -s, --seed (optional): Seed value for random number generation

REM Run the model using the inputs sampled above, and save outputs
python -c "from SALib.test_functions import Ishigami; import numpy as np; np.savetxt('../data/model_output.txt', Ishigami.evaluate(np.loadtxt('../data/model_input.txt')))"

REM Then use the output to run the analysis.
REM Sensitivity indices will print to command line. Use ">" to write to file.

salib analyze ff ^
  -p ../../SALib/test_functions/params/Ishigami.txt ^
  -Y ../data/model_output.txt ^
  -c 0 ^
  -X ../data/model_input.txt ^
  --seed=100

REM python -m SALib.analyze.ff ^
REM   -p ../../SALib/test_functions/params/Ishigami.txt ^
REM   -Y ../data/model_output.txt ^
REM   -c 0 ^
REM   -X ../data/model_input.txt ^
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
REM -s, --seed (optional): Seed value for random number generation
