==========================
Frequently Asked Questions
==========================


Q. How do I wrap my model?
---------------

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

    def wrapped_linear(abx):
        """f(X) = Y, where X := [a b x]"""
        # We transpose to obtain each column as separate variables
        a, b, x = abx.T

        # Then call the original model
        return linear(a, b, x)


Constants, which SALib should not consider, can be expressed by defining default keyword arguments.

.. code:: python

    def wrapped_linear_w_constant(bx, a=10):
        """f(X, a) = Y, where X := [b x] and a = 10"""
        # We transpose to obtain each column as separate variables
        b, x = bx.T

        # Then call the original model
        return linear(a, b, x)


Note that the wrapper's function arguments are arbitrarily named here. In actuality, they can
be given any name. The only requirement is that the matrix is the first argument. 
Using :py:func:`functools.partial` from the `functools` package to create wrappers can be useful.

In this example, the model can be used with both scalar inputs or `numpy` arrays. In cases where
`a`, `b` or `x` are a vector of inputs, `numpy` will automatically vectorize the calculation.

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

    # [         ST   ST_conf
    #  a  0.166667  0.006810
    #  b  0.166667  0.007519
    #  x  0.666669  0.026255,
    #           S1   S1_conf
    #  a  0.166667  0.016831
    #  b  0.166667  0.017186
    #  x  0.666669  0.031119,
    #                    S2   S2_conf
    #  (a, b)  2.328315e-09  0.025664
    #  (a, x) -3.813548e-06  0.031423
    #  (b, x) -3.815410e-06  0.032631]


Q. Which technique can I use for pre-existing results?
------------------------------------------------------

DMIM, RBD-FAST, PAWN and HDMR methods are "given-data" approaches and can be independently applied.


Q. Why does the Sobol' method implemented in SALib normalize outputs?
---------------------------------------------------------------------

Estimates of Sobol' indices can be biased in cases where model outputs are non-centered.
We have opted to normalize outputs with the standard deviation.

See the discussion here: https://github.com/SALib/SALib/issues/109#issuecomment-268499001

In practice, non-normalized model outputs are still usable but requires larger sample sizes for
the indices to converge.


Q. How do I get the sensitivity results?
----------------------------------------

Call the `.to_df()` method if you would like Pandas DataFrames.

If using the method chaining approach, see [the example here](https://github.com/SALib/SALib/tree/main/examples/Problem)


Q. How can I plot my results?
-----------------------------

SALib provides some basic plotting functionality. See [the example here](https://github.com/SALib/SALib/tree/main/examples/plotting).