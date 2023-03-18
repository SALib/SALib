Wrapping an existing model
--------------------------

SALib performs sensitivity analysis for any model that can be expressed in the form of :math:`f(X) = Y`,
where :math:`X` is a matrix of inputs (often referred to as the model's factors).

The analysis methods are independent of the model and can be applied non-intrusively such
that it does not matter what :math:`f` is.

Typical model implementations take the form of :math:`f(a, b, c, ...) = Y`. In other words, each model
factor is supplied as a separate argument to the function. In such cases it is necessary to
write a wrapper to allow use with SALib. This is illustrated here with a simple linear function:

.. code:: python

    def linear(a, b, x):
        """Return y = a + b + x"""
        return a + b + x

As SALib expects a (numpy) matrix of factors, we simply "wrap" the function above like so:

.. code:: python

    def wrapped_linear(X, func=linear):
        """g(X) = Y, where X := [a b x] and g(X) := f(X)"""
        # We transpose to obtain each column (the model factors) as separate variables
        a, b, x = X.T

        # Then call the original model
        return func(a, b, x)

.. note:: Wrapped function is an argument
    Note here that the model being "wrapped" is also passed in as an argument.
    This will be revisited further down below.


.. tip:: Interfacing with external models/programs
    Here we showcase interacting with models written in Python.
    If the model is an external program, this is where interfacing code
    would be written.

    A pragmatic approach could be to use `subprocess <https://docs.python.org/3/library/subprocess.html>`_
    to start the external program, then read in the results.


Constants, which SALib should not consider, can be expressed by defining default keyword arguments
(for flexibility) or otherwise defined within the wrapper function itself.

.. code:: python

    def wrapped_linear_w_constant(X, a=10, func=linear):
        """f(X, a) = Y, where X := [b x] and a = 10"""
        # We transpose to obtain each column as separate variables
        b, x = X.T

        # Then call the original model
        return func(a, b, x)


Note that the first argument to the wrapper function(s) is a numpy array of shape
:math:`N*D`, where :math:`D` is the number of model factors (dimensions) and
:math:`N` is the number of their combinations. The argument name is, by convention,
denoted as :code:`X`. This is to maximize compatibility with all methods provided
in SALib as they expect the first argument to hold the model factor values.
Using :py:func:`functools.partial` from the `functools` package to create wrappers can be useful.

In this example, the model (:code:`linear()`) can be used with both scalar inputs or `numpy` arrays.
In cases where `a`, `b` or `x` are a vector of inputs, `numpy` will automatically vectorize the
calculation.

There are many cases where the model is not (or cannot be easily) expressed in a vectorizable form.
When using the core SALib functions directly in such cases, the user is expected to evaluate the
model in a `for` loop themselves.

.. code:: python

    from SALib.sample import saltelli
    from SALib.analyze import sobol

    problem = {
        'names': ['a', 'b', 'x'],
        'bounds': [
            [-1, 0],
            [-1, 0],
            [-1, 1],
        ],
        'num_vars': 3
    }

    X = saltelli.sample(problem, 64)
    Y = np.empty(params.shape[0])
    for i in range(params.shape[0]):
        Y[i] = wrapped_linear(params[i, :])

    res = sobol.analyze(problem, Y)
    res.to_df()

    # [         ST   ST_conf
    #  a  0.165854  0.054096
    #  b  0.165854  0.053200
    #  x  0.665366  0.192756,
    #           S1   S1_conf
    #  a  0.167805  0.121550
    #  b  0.167805  0.125178
    #  x  0.665366  0.230872,
    #                    S2   S2_conf
    #  (a, b) -2.775558e-17  0.180493
    #  (a, x) -3.902439e-03  0.202343
    #  (b, x) -3.902439e-03  0.232957]

This highlights one usability aspect of using the SALib `ProblemSpec` Interface - it
automatically applies the model for each individual sample set in a `for` loop
(at the cost of computational efficiency).

.. code:: python

    from SALib import ProblemSpec


    sp = ProblemSpec({
        'names': ['a', 'b', 'x'],
        'bounds': [
            [-1, 0],
            [-1, 0],
            [-1, 1],
        ],
    })

    (
        sp.sample_sobol(2**6)
        .evaluate(wrapped_linear)
        .analyze_sobol()
    )

    sp.to_df()

    # [         ST   ST_conf
    #  a  0.173636  0.072142
    #  b  0.167933  0.059599
    #  x  0.654566  0.208328,
    #           S1   S1_conf
    #  a  0.182788  0.111548
    #  b  0.179003  0.145714
    #  x  0.664727  0.241977,
    #                S2   S2_conf
    #  (a, b) -0.022070  0.185510
    #  (a, x) -0.010781  0.186743
    #  (b, x) -0.014616  0.279925]

We also noted earlier that the model being "wrapped" is also passed in as an argument.
This is to facilitate parallel evaluation, as the arguments to the wrapper
are passed on to workers. The approach works be using Python's
`mutable default argument <https://docs.python-guide.org/writing/gotchas/#mutable-default-arguments>`_
behavior.

Technical detail aside, defining the model this way allows the model to be evaluated in parallel:

.. code:: python

    from SALib import ProblemSpec


    sp = ProblemSpec({
        'names': ['a', 'b', 'x'],
        'bounds': [
            [-1, 0],
            [-1, 0],
            [-1, 1],
        ],
    })

    (
        sp.sample_sobol(2**6)
        .evaluate(wrapped_linear, nprocs=2)
        .analyze_sobol()
    )

    sp.to_df()

    # [         ST   ST_conf
    #  a  0.166372  0.064571
    #  b  0.164554  0.068605
    #  x  0.665150  0.191152,
    #           S1   S1_conf
    #  a  0.201450  0.152915
    #  b  0.165128  0.124578
    #  x  0.670300  0.254541,
    #                S2   S2_conf
    #  (a, b) -0.027733  0.178632
    #  (a, x) -0.068051  0.257325
    #  (b, x)  0.000958  0.257001]
