@echo off

REM Sensitivity indices will print to command line. Use ">" to write to file.
salib analyze hdmr ^
  -p ../../../src/SALib/test_functions/params/Ishigami.txt ^
  -X ../../data/case4_input.txt ^
  -Y ../../data/case4_output.txt ^
  -c 0 ^
  -g 1 ^
  -mor 2 ^
  -mit 100 ^
  -m 4 ^
  -K 1 ^
  -R 1000 ^
  -a 0.95 ^
  -lambda 0.05 ^
  -print 1

PAUSE
REM Then use the output to run the analysis.

REM You can also use the module directly through Python
REM python -m SALib.analyze.hdmr ^
REM      -p ../../../src/SALib/test_functions/params/Ishigami.txtt ^
REM      -X ../../data/case4_input.txt ^
REM      -Y ../../data/case4_output.txt ^
REM      -c 0 ^
REM      -g 1 ^
REM      -mor 2 ^
REM      -mit 100 ^
REM      -m 4 ^
REM      -K 20 ^
REM      -R 500 ^
REM      -a 0.95 ^
REM      -lambda 0.05 ^
REM      -print 1

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
REM -g, --graphics (optional): Whether to print out graphics or not (1,0). Default is 1.
REM
REM -mor, --maxorder (optional): Maximum order of HDMR expansion (1,2,3). Default is 2.
REM
REM -mit, --maxiter (optional): Maximum iteration number for backfitting algorithm (1-1000). Default is 100.
REM
REM -m, --m-int (optional): Number of B-spline intervals. Default is 2.
REM
REM -K, --K-bootstrap (optional): Number of bootstrap (1,2,3). Default is 20.
REM
REM -R, --R-subsample (optional): Subsample size. Default is N/2.
REM
REM -a, --alfa (optional): Confidence interval. Default is 0.95
REM
REM -lambda, --lambdax (optional): Regularization constant. Default is 0.01
REM
REM -print, --print-to-console (optional): Whether to print out results to the screen or not (1,0). Default is 1.
