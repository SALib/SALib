##Sensitivity Analysis Library (SALib)

Python implementations of commonly used sensitivity analysis methods. Useful in systems modeling to calculate the effects of model inputs or exogenous factors on outputs of interest.

**Requirements:** [NumPy](http://www.numpy.org/), [SciPy](http://www.scipy.org/)

**Installation:** `pip install SALib` or `python setup.py install`

**Build Status:** [![Build Status](https://travis-ci.org/jdherman/SALib.svg?branch=master)](https://travis-ci.org/jdherman/SALib)    **Test Coverage:** [![Coverage Status](https://img.shields.io/coveralls/jdherman/SALib.svg)](https://coveralls.io/r/jdherman/SALib)

**Methods included:**
* Sobol Sensitivity Analysis ([Sobol 2001](http://www.sciencedirect.com/science/article/pii/S0378475400002706), [Saltelli 2002](http://www.sciencedirect.com/science/article/pii/S0010465502002801), [Saltelli et al. 2010](http://www.sciencedirect.com/science/article/pii/S0010465509003087))
* Method of Morris, including groups and optimal trajectories ([Morris 1991](http://www.tandfonline.com/doi/abs/10.1080/00401706.1991.10484804), [Campolongo et al. 2007](http://www.sciencedirect.com/science/article/pii/S1364815206002805))
* Fourier Amplitude Sensitivity Test (FAST) ([Cukier et al. 1973](http://scitation.aip.org/content/aip/journal/jcp/59/8/10.1063/1.1680571), [Saltelli et al. 1999](http://amstat.tandfonline.com/doi/abs/10.1080/00401706.1999.10485594))
* Delta Moment-Independent Measure ([Borgonovo 2007](http://www.sciencedirect.com/science/article/pii/S0951832006000883), [Plischke et al. 2013](http://www.sciencedirect.com/science/article/pii/S0377221712008995))
* Derivative-based Global Sensitivity Measure (DGSM) ([Sobol and Kucherenko 2009](http://www.sciencedirect.com/science/article/pii/S0378475409000354))

**Contributing:** see [here](CONTRIBUTING.md)

### Quick Start
```python
from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np

problem = {
  'num_vars': 3, 
  'names': ['x1', 'x2', 'x3'], 
  'bounds': [[-3.14159265359, 3.14159265359], 
            [-3.14159265359, 3.14159265359], 
             [-3.14159265359, 3.14159265359]]
}

# Generate samples
param_values = saltelli.sample(problem, 1000, calc_second_order=True)

# Run model (example)
Y = Ishigami.evaluate(param_values)
# for offline models, save param_values to a file:
# np.savetxt('model_input.txt', param_values, delimiter=' ')
# then load the model outputs with np.loadtxt()

# Perform analysis
Si = sobol.analyze(problem, Y, print_to_console=False)
# Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf'
# (first and total-order indices with bootstrap confidence intervals)
```

It's also possible to specify the parameter bounds in a file. Parameter files should be created with 3 columns:
```
# name lower_bound upper_bound
P1 0.0 1.0
P2 0.0 5.0
...etc.
```

Lots of other options are included for parameter files, as well as a command-line interface--see the [Advanced Readme](README-advanced.md).

Also check out the [examples](https://github.com/jdherman/SALib/tree/master/examples) for a full description of options for each method.

### License
Copyright (C) 2013-2015 Jon Herman and others. Licensed under the GNU Lesser General Public License.

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
