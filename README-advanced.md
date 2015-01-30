### SALib: Advanced options

#### Parameter files

In the parameter file, lines beginning with `#` will be treated as comments and ignored. The Morris method also supports groups of input factors, which can be specified with a fourth column:
```
# name lower_bound upper_bound group_name
P1 0.0 1.0 Group_1
P2 0.0 5.0 Group_2
P3 0.0 5.0 Group_2
...etc.
```
Parameter files can also be comma-delimited if your parameter names or group names contain spaces. This should be detected automatically.

#### Command-line interface

**Generate samples** (the `-p` flag is the parameter file)
```
python -m SALib.sample.saltelli \
     -n 1000 \
     -p ./SALib/test_functions/params/Ishigami.txt \
     -o model_input.txt \
```

**Run the model** this will usually be a user-defined model, maybe even in another language. Just save the outputs.

**Run the analysis**
```
python -m SALib.analyze.sobol \
     -p ./SALib/test_functions/params/Ishigami.txt \
     -Y model_output.txt \
     -c 0 \
```

This will print indices and confidence intervals to the command line. You can redirect to a file using the `>` operator.

Other methods include Morris, FAST, Delta-MIM, and DGSM. For an explanation of all command line options for each method, [see the examples here](https://github.com/jdherman/SALib/tree/master/examples).