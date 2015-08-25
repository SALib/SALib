##Sensitivity Analysis Library (SALib)

Python implementations of commonly used sensitivity analysis methods. Useful in systems modeling to calculate the effects of model inputs or exogenous factors on outputs of interest.

**Requirements:** [NumPy](http://www.numpy.org/), [SciPy](http://www.scipy.org/), [matplotlib](http://matplotlib.org/)

**Installation:** `pip install SALib` or `python setup.py install`

**Build Status:** [![Build Status](https://travis-ci.org/SALib/SALib.svg?branch=master)](https://travis-ci.org/SALib/SALib)    **Test Coverage:** [![Coverage Status](https://img.shields.io/coveralls/SALib/SALib.svg)](https://coveralls.io/r/SALib/SALib)

**Cite SALib:** [![DOI](https://zenodo.org/badge/15666/SALib/SALib.svg)](https://zenodo.org/badge/latestdoi/15666/SALib/SALib)

**Methods included:**
* Sobol Sensitivity Analysis ([Sobol 2001](http://www.sciencedirect.com/science/article/pii/S0378475400002706), [Saltelli 2002](http://www.sciencedirect.com/science/article/pii/S0010465502002801), [Saltelli et al. 2010](http://www.sciencedirect.com/science/article/pii/S0010465509003087))
* Method of Morris, including groups and optimal trajectories ([Morris 1991](http://www.tandfonline.com/doi/abs/10.1080/00401706.1991.10484804), [Campolongo et al. 2007](http://www.sciencedirect.com/science/article/pii/S1364815206002805))
* Fourier Amplitude Sensitivity Test (FAST) ([Cukier et al. 1973](http://scitation.aip.org/content/aip/journal/jcp/59/8/10.1063/1.1680571), [Saltelli et al. 1999](http://amstat.tandfonline.com/doi/abs/10.1080/00401706.1999.10485594))
* Delta Moment-Independent Measure ([Borgonovo 2007](http://www.sciencedirect.com/science/article/pii/S0951832006000883), [Plischke et al. 2013](http://www.sciencedirect.com/science/article/pii/S0377221712008995))
* Derivative-based Global Sensitivity Measure (DGSM) ([Sobol and Kucherenko 2009](http://www.sciencedirect.com/science/article/pii/S0378475409000354))
* Fractional Factorial Sensitivity Analysis ([Saltelli et al. 2008](http://www.wiley.com/WileyCDA/WileyTitle/productCd-0470059974.html))

**Contributing:** see [here](CONTRIBUTING.md)

### Quick Start
```python
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
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

# Perform analysis
Si = sobol.analyze(problem, Y, print_to_console=False)
# Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf'
# (first and total-order indices with bootstrap confidence intervals)
```

It's also possible to specify the parameter bounds in a file with 3 columns:
```
# name lower_bound upper_bound
P1 0.0 1.0
P2 0.0 5.0
...etc.
```

Then the `problem` dictionary above can be created from the `read_param_file` function:
```python
from SALib.util import read_param_file
problem = read_param_file('/path/to/file.txt')
# ... same as above
```

Lots of other options are included for parameter files, as well as a command-line interface. See the [advanced readme](README-advanced.md).

Also check out the [examples](https://github.com/SALib/SALib/tree/master/examples) for a full description of options for each method.

### License
Copyright (C) 2013-2015 Jon Herman, Will Usher, and others. Versions v0.5 and later are released under the [MIT license](LICENSE.md).
