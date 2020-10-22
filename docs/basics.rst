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

    f(x) = sin(x_1) + a sin^2(x_2) + b x_3^4 sin(x_1)
    
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

Next, we must define the model inputs.  The Ishigami function has three inputs,
:math:`x_1, x_2, x_3` where :math:`x_i \in [-\pi, \pi]`.  In SALib, we define
a :code:`dict` defining the number of inputs, the names of the inputs, and
the bounds on each input, as shown below.

.. code:: python

    problem = {
        'num_vars': 3, 
        'names': ['x1', 'x2', 'x3'], 
        'bounds': [[-3.14159265359, 3.14159265359], 
                   [-3.14159265359, 3.14159265359], 
                   [-3.14159265359, 3.14159265359]]
    }
    
Generate Samples
~~~~~~~~~~~~~~~~

Next, we generate the samples.  Since we are performing a Sobol' sensitivity
analysis, we need to generate samples using the Saltelli sampler, as shown
below.  

.. code:: python

    param_values = saltelli.sample(problem, 1000)
    
Here, :code:`param_values` is a NumPy matrix.  If we run
:code:`param_values.shape`, we see that the matrix is 8000 by 3.  The Saltelli
sampler generated 8000 samples.  The Saltelli sampler generates
:math:`N*(2D+2)` samples, where in this example N is 1000 (the argument we
supplied) and D is 3 (the number of model inputs). The keyword argument :code:`calc_second_order=False` will exclude second-order indices, resulting in a smaller sample matrix with :math:`N*(D+2)` rows instead.

Run Model
~~~~~~~~~

As mentioned above, SALib is not involved in the evaluation of the mathematical
or computational model.  If the model is written in Python, then generally you
will loop over each sample input and evaluate the model:

.. code:: python

    Y = np.zeros([param_values.shape[0]])

    for i, X in enumerate(param_values):
        Y[i] = evaluate_model(X)
        
If the model is not written in Python, then the samples can be saved to a text
file:

.. code:: python

    np.savetxt("param_values.txt", param_values)
    
Each line in :code:`param_values.txt` is one input to the model.  The output
from the model should be saved to another file with a similar format: one
output on each line.  The outputs can then be loaded with:

.. code:: python

    Y = np.loadtxt("outputs.txt", float)

In this example, we are using the Ishigami function provided by SALib.  We
can evaluate these test functions as shown below:

.. code:: python

    Y = Ishigami.evaluate(param_values)

Perform Analysis
~~~~~~~~~~~~~~~~

With the model outputs loaded into Python, we can finally compute the sensitivity
indices.  In this example, we use :code:`sobol.analyze`, which will compute
first, second, and total-order indices.

.. code:: python

    Si = sobol.analyze(problem, Y)
    
:code:`Si` is a Python :code:`dict` with the keys :code:`"S1"`,
:code:`"S2"`, :code:`"ST"`, :code:`"S1_conf"`, :code:`"S2_conf"`, and
:code:`"ST_conf"`.  The :code:`_conf` keys store the corresponding confidence
intervals, typically with a confidence level of 95%. Use the keyword argument :code:`print_to_console=True` to print all indices. Or, we can print the individual values from :code:`Si` as shown below.

.. code:: python

    print(Si['S1'])
    
    [ 0.30644324  0.44776661 -0.00104936 ]
    
Here, we see that x1 and x2 exhibit first-order sensitivities but x3 appears to
have no first-order effects.

.. code:: python

    print(Si['ST'])
    
    [ 0.56013728  0.4387225   0.24284474]

If the total-order indices are substantially larger than the first-order
indices, then there is likely higher-order interactions occurring.  We can look
at the second-order indices to see these higher-order interactions:

.. code:: python

    print "x1-x2:", Si['S2'][0,1]
    print "x1-x3:", Si['S2'][0,2]
    print "x2-x3:", Si['S2'][1,2]
    
    x1-x2: 0.0155279
    x1-x3: 0.25484902
    x2-x3: -0.00995392
    
We can see there are strong interactions between x1 and x3.  Some computing
error will appear in the sensitivity indices.  For example, we observe a
negative value for the x2-x3 index.  Typically, these computing errors shrink as
the number of samples increases.

The output can then be converted to a Pandas DataFrame for further analysis.

..code:: python
    total_Si, first_Si, second_Si = Si.to_df()

    # Note that if the sample was created with `calc_second_order=False`
    # Then the second order sensitivities will not be returned
    # total_Si, first_Si = Si.to_df()


     
Basic Plotting
~~~~~~~~~~~~~~~~

Basic plotting facilities are provided for convenience.

.. code:: python
    
    Si.plot()

The :code:`plot()` method returns matplotlib axes objects to allow later adjustment.
