@echo off

REM Sensitivity indices will print to command line. Use ">" to write to file.
salib analyze delta ^
  -p ../../SALib/test_functions/params/Ishigami.txt ^
  -X ../data/model_input.txt ^
  -Y ../data/model_output.txt ^
  -c 0 ^
  -r 10 ^
  --seed=100

REM Then use the output to run the analysis.

REM You can also use the module directly through Python
REM python -m SALib.analyze.delta ^
REM      -p ../../SALib/test_functions/params/Ishigami.txt ^
REM      -X ../data/model_input.txt ^
REM      -Y ../data/model_output.txt ^
REM      -c 0 ^
REM      -r 10 ^
REM      --seed=100

REM Options:
REM -p, --paramfile: Your parameter range file (3 columns: parameter name, lower bound, upper bound)
REM
REM -Y, --model-output-file: File of model output values to analyze
REM
REM -X, --model-input-file: File of model input values (parameter samples).
REM
REM -c, --column (optional): Column of model output file to analyze.
REM                If the file only has one column, this argument will be ignored.
REM
REM --delimiter (optional): Model output file delimiter.
REM
REM -r, --resamples (optional): Number of bootstrap resamples used to calculate confidence intervals on indices. Default 1000.
REM
REM -s, --seed (optional): Seed value for random number generation
