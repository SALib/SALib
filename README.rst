Sensitivity Analysis Library (SALib)
====================================

Python implementations of commonly used sensitivity analysis methods.
Useful in systems modeling to calculate the effects of model inputs or
exogenous factors on outputs of interest.

**Documentation:** `ReadTheDocs <http://salib.readthedocs.org>`__

**Requirements:** `NumPy <http://www.numpy.org/>`__,
`SciPy <http://www.scipy.org/>`__,
`matplotlib <http://matplotlib.org/>`__,
`pandas <http://https://pandas.pydata.org/>`__,
Python 3 (from SALib v1.2 onwards SALib does not officially support Python 2)

**Installation:** ``pip install SALib`` or ``python setup.py install`` or ``conda install SALib``

**Build Status:** |Build Status| **Test Coverage:** |Coverage Status|

Included methods
----------------

* Sobol Sensitivity Analysis (`Sobol 2001 <http://www.sciencedirect.com/science/article/pii/S0378475400002706>`__,
  `Saltelli 2002 <http://www.sciencedirect.com/science/article/pii/S0010465502002801>`__,
  `Saltelli et al. 2010 <http://www.sciencedirect.com/science/article/pii/S0010465509003087>`__)

* Method of Morris, including groups and optimal trajectories (`Morris
  1991 <http://www.tandfonline.com/doi/abs/10.1080/00401706.1991.10484804>`__,
  `Campolongo et al. 2007 <http://www.sciencedirect.com/science/article/pii/S1364815206002805>`__,
  `Ruano et al. 2012 <https://doi.org/10.1016/j.envsoft.2012.03.008>`__)

* extended Fourier Amplitude Sensitivity Test (eFAST) (`Cukier et al. 1973 <http://scitation.aip.org/content/aip/journal/jcp/59/8/10.1063/1.1680571>`__,
  `Saltelli et al. 1999 <http://amstat.tandfonline.com/doi/abs/10.1080/00401706.1999.10485594>`__, `Pujol (2006) in Iooss et al., (2021) <http://scitation.aip.org/content/aip/journal/jcp/59/8/10.1063/1.1680571>`__)

* Random Balance Designs - Fourier Amplitude Sensitivity Test (RBD-FAST) (`Tarantola et al. 2006 <https://hal.archives-ouvertes.fr/hal-01065897/file/Tarantola06RESS_HAL.pdf>`__,
  `Plischke 2010 <https://doi.org/10.1016/j.ress.2009.11.005>`__, 
  `Tissot et al. 2012 <https://doi.org/10.1016/j.ress.2012.06.010>`__) 

* Delta
  Moment-Independent Measure (`Borgonovo 2007 <http://www.sciencedirect.com/science/article/pii/S0951832006000883>`__,
  `Plischke et al. 2013 <http://www.sciencedirect.com/science/article/pii/S0377221712008995>`__)

* Derivative-based Global Sensitivity Measure (DGSM) (`Sobol and
  Kucherenko 2009 <http://www.sciencedirect.com/science/article/pii/S0378475409000354>`__)

* Fractional Factorial Sensitivity Analysis 
  (`Saltelli et al. 2008 <http://www.wiley.com/WileyCDA/WileyTitle/productCd-0470059974.html>`__)

* High-Dimensional Model Representation (HDMR) 
  (`Rabitz et al. 1999 <https://doi.org/10.1016/S0010-4655(98)00152-0>`__, `Li et al. 2010 <https://doi.org/10.1021/jp9096919>`__)

* PAWN (`Pianosi and Wagener 2018 <10.1016/j.envsoft.2018.07.019>`__, `Pianosi and Wagener 2015 <https://doi.org/10.1016/j.envsoft.2015.01.004>`__)


**Contributing:** see `here <CONTRIBUTING.md>`__

Quick Start
-----------

Procedural approach
~~~~~~~~~~~~~~~~~~~

.. code:: python

    from SALib.sample import saltelli
    from SALib.analyze import sobol
    from SALib.test_functions import Ishigami
    import numpy as np

    problem = {
      'num_vars': 3,
      'names': ['x1', 'x2', 'x3'],
      'bounds': [[-np.pi, np.pi]]*3
    }

    # Generate samples
    param_values = saltelli.sample(problem, 1024)

    # Run model (example)
    Y = Ishigami.evaluate(param_values)

    # Perform analysis
    Si = sobol.analyze(problem, Y, print_to_console=True)
    # Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf'
    # (first and total-order indices with bootstrap confidence intervals)

It's also possible to specify the parameter bounds in a file with 3
columns:

::

    # name lower_bound upper_bound
    P1 0.0 1.0
    P2 0.0 5.0
    ...etc.

Then the ``problem`` dictionary above can be created from the
``read_param_file`` function:

.. code:: python

    from SALib.util import read_param_file
    problem = read_param_file('/path/to/file.txt')
    # ... same as above

Lots of other options are included for parameter files, as well as a
command-line interface. See the `advanced
section in the documentation <https://salib.readthedocs.io/en/latest/advanced.html>`__.


Method chaining approach
~~~~~~~~~~~~~~~~~~~~~~~~

Chaining calls is supported from SALib v1.4 

.. code:: python

    from SALib import ProblemSpec
    from SALib.test_functions import Ishigami

    import numpy as np


    # By convention, we assign to "sp" (for "SALib Problem")
    sp = ProblemSpec({
      'names': ['x1', 'x2', 'x3'],   # Name of each parameter
      'bounds': [[-np.pi, np.pi]]*3,  # bounds of each parameter
      'outputs': ['Y']               # name of outputs in expected order
    })

    (sp.sample_saltelli(1024, calc_second_order=True)
       .evaluate(Ishigami.evaluate)
       .analyze_sobol(print_to_console=True))

    print(sp)

    # Samples, model results and analyses can be extracted:
    print(sp.samples)
    print(sp.results)
    print(sp.analysis)

    # Basic plotting functionality is also provided
    sp.plot()


The above is equivalent to the procedural approach shown previously.

Also check out the
`examples <https://github.com/SALib/SALib/tree/main/examples>`__ for a
full description of options for each method.


How to cite SAlib
-----------------

If you would like to use our software, please cite it using the following:

    Herman, J. and Usher, W. (2017) SALib: An open-source Python library for
    sensitivity analysis. Journal of Open Source Software, 2(9).
    doi:10.21105/joss.00097

|paper status|

If you use BibTeX, cite using the following entry::

    @article{Herman2017,
      doi = {10.21105/joss.00097},
      url = {https://doi.org/10.21105/joss.00097},
      year  = {2017},
      month = {jan},
      publisher = {The Open Journal},
      volume = {2},
      number = {9},
      author = {Jon Herman and Will Usher},
      title = {{SALib}: An open-source Python library for Sensitivity Analysis},
      journal = {The Journal of Open Source Software}
    }

Projects that use SALib
-----------------------

Many projects now use the Global Sensitivity Analysis features provided by
SALib. Here is a selection:

Software
~~~~~~~~

* `The City Energy Analyst <https://github.com/architecture-building-systems/CEAforArcGIS>`_
* `pynoddy <https://github.com/flohorovicic/pynoddy>`_
* `savvy <https://github.com/houghb/savvy>`_
* `rhodium <https://github.com/Project-Platypus/Rhodium>`_
* `pySur <https://github.com/MastenSpace/pysur>`_
* `EMA workbench <https://github.com/quaquel/EMAworkbench>`_
* `Brain/Circulation Model Developer <https://github.com/bcmd/BCMD>`_
* `DAE Tools <http://daetools.com/>`_
* `agentpy <https://github.com/JoelForamitti/agentpy>`_
* `uncertainpy <https://github.com/simetenn/uncertainpy>`_

Blogs
~~~~~

* `Sensitivity Analyis in Python <http://www.perrygeo.com/sensitivity-analysis-in-python.html>`_
* `Sensitivity Analysis with SALib <http://keyboardscientist.weebly.com/blog/sensitivity-analysis-with-salib>`_
* `Running Sobol using SALib <https://waterprogramming.wordpress.com/2013/08/05/running-sobol-sensitivity-analysis-using-salib/>`_
* `Extensions of SALib for more complex sensitivity analyses <https://waterprogramming.wordpress.com/2014/02/11/extensions-of-salib-for-more-complex-sensitivity-analyses/>`_

Videos
~~~~~~

* `PyData Presentation on SALib <https://youtu.be/gkR_lz5OptU>`_

If you would like to be added to this list, please submit a pull request,
or create an issue.

Many thanks for using SALib.


How to contribute
-----------------

See `here <CONTRIBUTING.md>`__ for how to contribute to SAlib.


License
-------

Copyright (C) 2012-2019 Jon Herman, Will Usher, and others. Versions v0.5 and
later are released under the `MIT license <LICENSE.md>`__.

.. |Build Status| image:: https://travis-ci.com/SALib/SALib.svg?branch=master
   :target: https://travis-ci.com/SALib/SALib
.. |Coverage Status| image:: https://img.shields.io/coveralls/SALib/SALib.svg
   :target: https://coveralls.io/r/SALib/SALib
.. |Code Issues| image:: https://www.quantifiedcode.com/api/v1/project/ed62e70f899e4ec8af4ea6b2212d4b30/badge.svg
   :target: https://www.quantifiedcode.com/app/project/ed62e70f899e4ec8af4ea6b2212d4b30
.. |paper status| image:: http://joss.theoj.org/papers/431262803744581c1d4b6a95892d3343/status.svg
   :target: http://joss.theoj.org/papers/431262803744581c1d4b6a95892d3343
