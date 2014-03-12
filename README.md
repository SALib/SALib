##Sensitivity Analysis Library
####Python implementations of Sobol, Method of Morris, and FAST sensitivity analysis methods

This library provides sampling and analysis routines for commonly used sensitivity analysis methods. These are typically applied in model diagnostics to understand the effects of model parameters on outputs of interest. Requires [NumPy](http://www.numpy.org/).

To get started, create a file describing the sampling ranges for the parameters in the model. Parameter files should be created with 3 columns, name, lower bound, and upper bound, e.g.:
```
P1 0.0 1.0
P2 0.0 5.0
...etc.
```

If the parameters are to be sampled with normal distributions, the lines in the parameter file should read (name, mean, standard deviation). None of the three methods uses normal sampling, but it is included for other user-defined applications.

There are two ways to run the library: as a decoupled process from the command line, or from a Python script.

### Command-Line Interface

For most applications, it will be helpful to run three separate processes: sampling, evaluation, and analysis. This approach is strongly recommended for users whose applications do not require modifications to the sampling/analysis routines. Steps:

**Step 1:** Perform sampling (see `shell-example-sample.sh` for an example). Choose a method and a number of samples, along with other important options as shown in the file.
```
python -B -m SALib.sample \
	   -m saltelli \
	   -n 100 \
	   -p ./SALib/test_functions/params/Sobol_G.txt \
	   -o my_samples_file.txt \
	   --delimiter=' ' \
	   --precision=8 \
```

Note that `python -m` runs the `__main__` module in the `SALib.sample` package. The `-B` flag prevents the interpreter from compiling bytecode (`.pyc` files). Options include:

* `-p, --paramfile`: Required. Your parameter range file (3 columns: parameter name, lower bound, upper bound).

* `-m, --method`: Required. Choose one of `{uniform, normal, latin, saltelli, morris, fast}`. All methods except `normal` assume the ranges are (lower bound, upper bound); `normal` assumes the ranges are (mean, standard deviation). The methods `saltelli`, `morris`, and `fast` correspond to the analysis methods `sobol`, `morris`, and `fast`, respectively. 

* `-n, --samples`: Sample size (required). The sample sizes generated for each method are as follows (where `D` is the number of parameters and `N` is the sample size):
	* Uniform: `N`
	* Normal: `N`
	* Latin: `N`
	* Saltelli: `N(2D + 2)` if calculating second-order indices (default) or `N(D + 2)` otherwise.						
	* Morris: `N(D + 1)`
	* FAST: `ND` ; must choose `N > 4M^2` (`N > 64` under default settings)

* `-o, --output`: File to output your samples into (required).

* `-s, --seed`: Random seed (optional). Will not affect Sobol method as it uses quasi-random sampling.
 
* `--delimiter`: Output file delimiter (optional). Common choices:
	* Space-delimited (default): `--delimiter=' '`
	* Comma-delimited: `--delimiter=','`
	* Tab-delimited: `--delimiter=$'\t'`

* `--precision`: Digits of precision in the output file (optional). Default is 8.

* `--saltelli-max-order`: Maximum order of indices to calculate (optional). Only applies for `-m saltelli`. Choose 1 or 2 (default is 2). 
	* Choosing 1 will reduce total model runs from `N(2D + 2)` to `N(D + 2)`
	* Must use the same value (either 1 or 2) for both sampling and analysis.

* `--morris-num-levels`: Number of levels in the OAT sampling (optional). The range of each variable will be discretized into this many levels. Only applies for `-m morris`. Default is 10.

* `--morris-grid-jump`: Grid jump size in the OAT sampling (optional). Each variable will be perturbed by this number of levels during each trajectory. Only applies for `-m morris`. Default is 5.

The parameter samples will be saved in the file specified by the `-o` flag. The file will contain `D` columns, one for each parameter.


**Step 2:** Perform model evaluations using the file of samples. This occurs independently of this library, other than the fact that you can specify a delimiter during sampling which might simplify the process of reading the parameter samples into your model. Save model output(s) of interest to a file and proceed.

**Step 3:** Perform analysis of the model output (see `shell-analyze-sample.sh` for an example). The analysis routines return measures of parameter sensitivity. It will print to the terminal by default; you may want to redirect output to a file using the `>` operator.
```
python -B -m SALib.analyze \
	   -m sobol \
	   -p ./SALib/test_functions/params/Sobol_G.txt \
	   -Y SGOutput.txt \
	   -c 0 \
```
	   
Options:
* `-p, --paramfile`: Required. Your parameter range file (3 columns: parameter name, lower bound, upper bound).

* `-m, --method`: Required. Choose one of `{sobol, morris, fast}`. *Must correspond to the method chosen during sampling*.

* `-Y, --model-output-file`: Required. File of model output values to analyze.

* `-c, --column`: Column of model output file to analyze (optional, default is zero). If the file only has one column, this argument will be ignored.

* `--delimiter`: Model output file delimiter (optional). Common choices:
	* Space-delimited (default): `--delimiter=' '`
	* Comma-delimited: `--delimiter=','`
	* Tab-delimited: `--delimiter=$'\t'`

* `--sobol-max-order`: Maximum order of indices to calculate (optional). Only applies for `-m sobol`. Choose 1 or 2 (default is 2). *This must match the value chosen during sampling*.

* `-X, --morris-model-input`: Required for Method of Morris only. File of model input values (parameter samples).

* `-r, --sobol-bootstrap-resamples`: Number of bootstrap resamples used to calculate confidence intervals on Sobol indices (optional).


### Python Interface
The library can also be used directly from a Python script. This approach has more of a learning curve and is only recommended for users who need to customize sampling and/or analysis processes for their applications. Refer to `python-example.py` for an example of how each of the methods are invoked from Python.

### License
Copyright (C) 2013 Jon Herman, Patrick Reed and others. Licensed under the GNU Lesser General Public License.

The Sensitivity Analysis Library is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The Sensitivity Analysis Library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the Sensitivity Analysis Library.  If not, see <http://www.gnu.org/licenses/>.