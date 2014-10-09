##Sensitivity Analysis Library (SALib)

Python implementations of commonly used sensitivity analysis methods. Useful in systems modeling to calculate the effects of model inputs or exogenous factors on outputs of interest. 

**Requirements:** [NumPy](http://www.numpy.org/), [SciPy](http://www.scipy.org/)

**Methods included:**
* Sobol Sensitivity Analysis ([Sobol 2001](http://www.sciencedirect.com/science/article/pii/S0378475400002706), [Saltelli 2002](http://www.sciencedirect.com/science/article/pii/S0010465502002801), [Saltelli et al. 2010](http://www.sciencedirect.com/science/article/pii/S0010465509003087))
* Method of Morris ([Morris 1991](http://www.tandfonline.com/doi/abs/10.1080/00401706.1991.10484804), [Campolongo et al. 2007](http://www.sciencedirect.com/science/article/pii/S1364815206002805))
* Fourier Amplitude Sensitivity Test (FAST) ([Cukier et al. 1973](http://scitation.aip.org/content/aip/journal/jcp/59/8/10.1063/1.1680571), [Saltelli et al. 1999](http://amstat.tandfonline.com/doi/abs/10.1080/00401706.1999.10485594))
* Delta Moment-Independent Measure ([Borgonovo 2007](http://www.sciencedirect.com/science/article/pii/S0951832006000883), [Plischke et al. 2013](http://www.sciencedirect.com/science/article/pii/S0377221712008995))
* Derivative-based Global Sensitivity Measure (DGSM) ([Sobol and Kucherenko 2009](http://www.sciencedirect.com/science/article/pii/S0378475409000354))
* Metamodel-based Sobol Analysis (experimental). Uses RBF support vector regression from `scikit-learn`.

**Contributors:** [Jon Herman](https://github.com/jdherman), [Matt Woodruff](https://github.com/matthewjwoodruff), [Chris Mutel](https://github.com/cmutel) [Fernando Rios](https://github.com/zoidy), [Dan Hyams](https://github.com/dhyams)

### Create a parameter file

To get started, create a file describing the sampling ranges for the parameters in the model. Parameter files should be created with 3 columns: `[name, lower bound, upper bound]`:
```
P1 0.0 1.0
P2 0.0 5.0
...etc.
```

Lines beginning with `#` will be treated as comments and ignored.

### Generate samples

From the command line:
```
python -m SALib.sample.saltelli \
     -n 1000 \
     -p ./SALib/test_functions/params/Ishigami.txt \
     -o model_input.txt \
```

Other methods include `SALib.sample.morris_oat` and `SALib.sample.fast_sampler`. For an explanation of all command line options, [see the examples here](https://github.com/jdherman/SALib/tree/master/examples). 

Or, generate samples from Python:
```python
from SALib.sample import saltelli
import numpy as np

param_file = '../../SALib/test_functions/params/Ishigami.txt'
param_values = saltelli.sample(1000, param_file, calc_second_order = True)
np.savetxt('model_input.txt', param_values, delimiter=' ')
```

Either way, this will create a file of sampled input values in `model_input.txt`.

### Run model
Here's an example of running a test function in Python, but this will usually be a user-defined model, maybe even in another language. Just save the outputs.

```python
from SALib.test_functions import Ishigami
Y = Ishigami.evaluate(param_values)
np.savetxt('model_output.txt', Y, delimiter=' ')
```

### Analyze model output

From the command line:
```
python -m SALib.analyze.sobol \
     -p ./SALib/test_functions/params/Ishigami.txt \
     -Y model_output.txt \
     -c 0 \
```

This will print indices and confidence intervals to the command line. You can redirect to a file using the `>` operator.

Or, from Python:
```python
from SALib.analyze import sobol
import numpy as np
Si = sobol.analyze(param_file, 'model_output.txt', column = 0, print_to_console=False)
# Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf'
# e.g. Si['S1'] contains the first-order index for each parameter, in the same order as the parameter file
```
	  
Check out the [examples](https://github.com/jdherman/SALib/tree/master/examples) for a full description of command line and keyword options for all of the methods.


### License
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