# FAQ

## Installation

Q. I get this error when installing SALib 1.3 with Python 2.7
A. SALib 1.3 onwards does not support Python 2.7


## Usage

Q. I've already got results from my Monte Carlo simulation - which technique can I use?
A. DMIM, RBD-FAST, and HDMR methods are sampling scheme independent.

Q. Can you help me with implementing my sensitivity analysis?
A. Check out [the examples here](https://github.com/SALib/SALib/tree/master/examples)

Q. The example(s) use the Ishigami function. How do I use it with my own model?
A.

Q. How do I get my results?
A. Call the `.to_df()` method if you would like Pandas DataFrames.
   The numpy array can be obtained via `.analysis`

Q. How can I plot my results?
A. SALib provides some basic plotting functionality. See [the example here](https://github.com/SALib/SALib/tree/master/examples/plotting).



