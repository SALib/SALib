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
sensitivity analyses.  SALib provides a decoupled workflow, meaning it does not
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
function (shown below) using the core SALib functions. The example is repeated
in the next tutorial using an object-oriented interface which some may find
easier to use.

The Ishigami function is commonly used to test uncertainty and sensitivity
analysis methods because it exhibits strong nonlinearity and nonmonotonicity.

.. math::

    f(x) = \sin(x_1) + a \sin^2(x_2) + b x_3^4 \sin(x_1)

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

    param_values = saltelli.sample(problem, 1024)

Here, :code:`param_values` is a NumPy matrix.  If we run
:code:`param_values.shape`, we see that the matrix is 8000 by 3.  The Saltelli
sampler generated 8000 samples.  The Saltelli sampler generates
:math:`N*(2D+2)` samples, where in this example N is 1024 (the argument we
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

    [ 0.316832  0.443763 0.012203 ]

Here, we see that x1 and x2 exhibit first-order sensitivities but x3 appears to
have no first-order effects.

.. code:: python

    print(Si['ST'])

    [ 0.555860  0.441898   0.244675]

If the total-order indices are substantially larger than the first-order
indices, then there is likely higher-order interactions occurring.  We can look
at the second-order indices to see these higher-order interactions:

.. code:: python

    print("x1-x2:", Si['S2'][0,1])
    print("x1-x3:", Si['S2'][0,2])
    print("x2-x3:", Si['S2'][1,2])

    x1-x2: 0.0092542
    x1-x3: 0.2381721
    x2-x3: -0.0048877

We can see there are strong interactions between x1 and x3.  Some computing
error will appear in the sensitivity indices.  For example, we observe a
negative value for the x2-x3 index.  Typically, these computing errors shrink as
the number of samples increases.

The output can then be converted to a Pandas DataFrame for further analysis.

.. code:: python

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


Another Example
---------------

When the model you want to analyse depends on parameters that are not part of
the sensitivity analysis, like position or time, the analysis can be performed
for each time/position "bin" separately.

Consider the example of a parabola:

.. math::

    f(x) = a + b x^2

The parameters :math:`a` and :math:`b` will be subject to the sensitivity analysis,
but :math:`x` will be not.

We start with a set of imports:

.. code:: python

    import numpy as np
    import matplotlib.pyplot as plt

    from SALib.sample import saltelli
    from SALib.analyze import sobol

and define the parabola:

.. code:: python

    def parabola(x, a, b):
        """Return y = a + b*x**2."""
        return a + b*x**2

The :code:`dict` describing the problem contains therefore only :math:`a` and :math:`b`:

.. code:: python

    problem = {
        'num_vars': 2,
        'names': ['a', 'b'],
        'bounds': [[0, 1]]*2
    }

The triad of sampling, evaluating and analysing becomes:

.. code:: python

    # sample
    param_values = saltelli.sample(problem, 2**6)

    # evaluate
    x = np.linspace(-1, 1, 100)
    y = np.array([parabola(x, *params) for params in param_values])

    # analyse
    sobol_indices = [sobol.analyze(problem, Y) for Y in y.T]

Note how we analysed for each :math:`x` separately.

Now we can extract the first-order Sobol indices for each bin of :math:`x` and plot:

.. code:: python

    S1s = np.array([s['S1'] for s in sobol_indices])

    fig = plt.figure(figsize=(10, 6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])

    for i, ax in enumerate([ax1, ax2]):
        ax.plot(x, S1s[:, i],
                label=r'S1$_\mathregular{{{}}}$'.format(problem["names"][i]),
                color='black')
        ax.set_xlabel("x")
        ax.set_ylabel("First-order Sobol index")

        ax.set_ylim(0, 1.04)

        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()

        ax.legend(loc='upper right')

    ax0.plot(x, np.mean(y, axis=0), label="Mean", color='black')

    # in percent
    prediction_interval = 95

    ax0.fill_between(x,
                     np.percentile(y, 50 - prediction_interval/2., axis=0),
                     np.percentile(y, 50 + prediction_interval/2., axis=0),
                     alpha=0.5, color='black',
                     label=f"{prediction_interval} % prediction interval")

    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.legend(title=r"$y=a+b\cdot x^2$",
               loc='upper center')._legend_box.align = "left"

    plt.show()

.. figure:: ../assets/example_parabola.svg
    :width: 800
    :align: center

With the help of the plots, we interpret the Sobol indices. At
:math:`x=0`, the variation in :math:`y` can be explained to 100 % by
parameter :math:`a` as the contribution to :math:`y` from :math:`b
x^2` vanishes. With larger :math:`|x|`, the contribution to the
variation from parameter :math:`b` increases and the contribution from
parameter :math:`a` decreases.
