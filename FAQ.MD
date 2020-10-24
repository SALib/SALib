# Frequently Asked Questions

## Installation

### I get this error when installing SALib 1.3 with Python 2.7
SALib 1.3 onwards does not support Python 2.7


## Usage

### I've already got results from my Monte Carlo simulation - which technique can I use?
DMIM, RBD-FAST, PAWN and HDMR methods are sampling scheme independent.


### Can you help me with implementing my sensitivity analysis?
Check out [the examples here](https://github.com/SALib/SALib/tree/develop/examples)


### The example(s) use the Ishigami function. How do I use SALib with my own model?
Two options:

1. Use SALib to generate the samples, run the model yourself, then use the
   results with SALib.
2. Wrap the model with a Python function. The only requirement is that the 
   function should accept a numpy array of sample values.
   This is the approach most users will take.


### How do I get the sensitivity results?
Call the `.to_df()` method if you would like Pandas DataFrames.

If using the method chaining approach, see [the example here](https://github.com/SALib/SALib/tree/develop/examples/problem)


### How can I plot my results?
SALib provides some basic plotting functionality. See [the example here](https://github.com/SALib/SALib/tree/develop/examples/plotting).
