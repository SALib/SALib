======
Basics
======

What is Sensitivity Analysis?
-----------------------------

According to `Wikipedia <https://en.wikipedia.org/wiki/Sensitivity_analysis>`_,
sensitivity analysis is "the study of how the uncertainty in the output of a 
mathematical model or system (numerical or otherwise) can be apportioned to 
different sources of uncertainty in its inputs."  The sensitivity of each input 
is often represented by a numeric value, called the *sensitivity index*.  
Sensitivity indices come in several forms:

1. First-order indices: measures the contribution to the output variance by a single
   model input alone.
   
2. Second-order indices: measures the contribution to the output variance caused by
   the interaction of two model inputs.
   
3. Total-order index: measures the contribution to the output variance caused by
   a model input, including both its first-order effects (the input varying alone)
   and all higher-order interactions.
   
What is SALib?
--------------

SALib is an open source library written in Python for performing
sensitivity analysis.  SALib provides a decoupled workflow, meaning it does not
directly interface with the mathematical or computational model.  Instead,
SALib is responsible for generating the model inputs, using one of the
:code:`sample` functions, and computing the sensitivity indices from the model
outputs, using one of the :code:`analyze` functions.  A typical sensitivity 
analysis using SALib follows four steps:

1. Determine the model inputs (parameters) and their sample range.  

2. Run the :code:`sample` function to generate the model inputs.

3. Evaluate the model using the generated inputs, saving the model outputs.

4. Run the :code:`analyze` function on the outputs to compute the sensitivity indices.

SALib provides several sensitivity analysis methods, such as Sobol, Morris,
and FAST.  There are many factors that determine which method is appropriate
for a specific application, which we will discuss later.  However, for now, just
remember that regardless of which method you choose, you need to use only two
functions: :code:`sample` and :code:`analyze`.  To demonstrate the use of SALib,
we will walk you through a simple example.

An Example
----------
In this example, we will perform a Sobol' sensitivity analysis of the Ishigami 
function, shown below.  The Ishigami function is commonly used to test 
uncertainty and sensitivity analysis methods because it exhibits strong 
nonlinearity and nonmonotonicity.

.. math::

    f(x) = sin(x_1) + a sin^2(X_2) + b x_3^4*sin(x_1)
    
Importing SALib
~~~~~~~~~~~~~~~

The first step is the import the necessary libraries.  In SALib, the
:code:`sample` and :code:`analyze` functions are stored in separate
Python modules.  For example, below we import the :code:`saltelli` sample
function and the :code:`sobol` analyze function.  We also import the Ishigami
function, which is provided as a test function within SALib.  Lastly, we
import :code:`numpy`, as it is used by SALib to store the model inputs and
outputs in a matrix.

.. code:: python

    from SALib.sample import saltelli
    from SALib.analyze import sobol
    from SALib.test_functions import Ishigami
    import numpy as np
    
Defining the Model Inputs
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    # Define the model inputs
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

    # Print the first-order sensitivity indices
    print Si['S1']

.. autofunction:: SALib.analyze.fast.analyze

