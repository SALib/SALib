==================
Advanced Examples
==================

Group sampling (Sobol and Morris methods only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
It is sometimes useful to perform sensitivity analysis on groups of input variables to reduce the number of model runs required, when variables belong to the same component of a model, or there is some reason to believe that they should behave similarly.

Groups can be specified in two ways for the Sobol and Morris methods. 

First, as a fourth column in the parameter file:

.. code:: python

    # name lower_bound upper_bound group_name
    P1 0.0 1.0 Group_1
    P2 0.0 5.0 Group_2
    P3 0.0 5.0 Group_2
    ...etc.

Or in the `problem` dictionary:

.. code:: python

    problem = {
        'groups': ['Group_1', 'Group_2', 'Group_2'],
        'names': ['x1', 'x2', 'x3'],
        'num_vars': 3,
        'bounds': [[-3.14, 3.14], [-3.14, 3.14], [-3.14, 3.14]]
    }

:code:`groups` is a list of strings specifying the group name to which each variable belongs. The rest of the code stays the same:

.. code:: python

    param_values = saltelli.sample(problem, 1000)
    Y = Ishigami.evaluate(param_values)
    Si = sobol.analyze(problem, Y, print_to_console=True)


But the output is printed by group:
::

    Group S1 S1_conf ST ST_conf
    Group_1 0.307834 0.066424 0.559577 0.082978
    Group_2 0.444052 0.080255 0.667258 0.060871

    Group_1 Group_2 S2 S2_conf
    Group_1 Group_2 0.242964 0.124229


Generating alternate distributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In :ref:`basics`, we generate a uniform sample of parameter space.

.. code:: python

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

    param_values = saltelli.sample(problem, 1000)

SALib is also capable of generating alternate sampling distributions by 
specifying a :code:`dists` entry in the :code:`problem` specification.

As implied in the basic example, a uniform distribution is the default.

When an entry for :code:`dists` is not 'unif', the :code:`bounds` entry does not indicate
parameter bounds but sample-specific metadata.

:code:`bounds` definitions for available distributions:

* unif: uniform distribution
    e.g. :code:`[-np.pi, np.pi]` defines the lower and upper bounds
* triang: triangular with width (scale) and location of peak. 
    Location of peak is in percentage of width.
    Lower bound assumed to be zero.

    e.g. :code:`[3, 0.5]` assumes 0 to 3, with a peak at 1.5
* norm: normal distribution with mean and standard deviation
* lognorm: lognormal with ln-space mean and standard deviation


An example specification is shown below:

.. code:: python

    problem = {
        'names': ['x1', 'x2', 'x3'],
        'num_vars': 3,
        'bounds': [[-np.pi, np.pi], [1.0, 0.2], [3, 0.5]],
        'groups': ['G1', 'G2', 'G1'],
        'dists': ['unif', 'lognorm', 'triang']
    }


