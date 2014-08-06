###Sensitivity Analysis Library (SALib)

Python implementations of commonly used sensitivity analysis methods. Useful in systems modeling to calculate the effects of model inputs or exogenous factors on outputs of interest. 

**Requirements:** [NumPy](http://www.numpy.org/), [SciPy](http://www.scipy.org/)

**Methods included:**
* Sobol Sensitivity Analysis ([Sobol 2001](http://www.sciencedirect.com/science/article/pii/S0378475400002706), [Saltelli 2002](http://www.sciencedirect.com/science/article/pii/S0010465502002801), [Saltelli et al. 2008](http://www.wiley.com/WileyCDA/WileyTitle/productCd-0470059974.html))
* Method of Morris ([Morris 1991](http://www.tandfonline.com/doi/abs/10.1080/00401706.1991.10484804), [Campolongo et al. 2007](http://www.sciencedirect.com/science/article/pii/S1364815206002805))
* Fourier Amplitude Sensitivity Test (FAST) ([Cukier et al. 1973](http://scitation.aip.org/content/aip/journal/jcp/59/8/10.1063/1.1680571), [Saltelli et al. 1999](http://amstat.tandfonline.com/doi/abs/10.1080/00401706.1999.10485594))

**Contributors:** [Jon Herman](https://github.com/jdherman), [Matt Woodruff](https://github.com/matthewjwoodruff), [Fernando Rios](https://github.com/zoidy), [Dan Hyams](https://github.com/dhyams)

#### Create a parameter file

To get started, create a file describing the sampling ranges for the parameters in the model. Parameter files should be created with 3 columns, name, lower bound, and upper bound, e.g.:
```
P1 0.0 1.0
P2 0.0 5.0
...etc.
```

Note that lines beginning with `#` will be treated as comments and ignored.

#### Generate sample

From the **command line**:
```
python -m SALib.sample.saltelli \
     -n 1000 \
     -p ./SALib/test_functions/params/Ishigami.txt \
     -o model_input.txt \
```

Other methods include `SALib.sample.morris_oat` and `SALib.sample.fast_sampler`. For an explanation of all command line options, [see the examples here](https://github.com/jdherman/SALib/tree/master/examples). 

Or, generate samples **from Python**:
```python
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np

# Read the parameter range file and generate samples
param_file = '../../SALib/test_functions/params/Ishigami.txt'

# Generate samples
param_values = saltelli.sample(1000, param_file, calc_second_order = True)
np.savetxt('model_input.txt', param_values, delimiter=' ')
```

Either way, this will create a file of sampled input values in `model_input.txt`.




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


#### Python Interface
The library can also be used directly from a Python script. This approach has more of a learning curve and is only recommended for users who need to customize sampling and/or analysis processes for their applications. Refer to `python-example.py` for an example of how each of the methods are invoked from Python. Sensitivity indices are printed to the command line but can also be returned in a dictionary.







https://github.com/jdherman/SALib/tree/feature/cleanup-structure/examples



#### License
Copyright (C) 2013-2014 Jon Herman and others. Licensed under the GNU Lesser General Public License.

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