## SALib: Advanced options

### Parameter files

In the parameter file, lines beginning with `#` will be treated as comments and ignored.
```
# name lower_bound upper_bound
P1 0.0 1.0
P2 0.0 5.0
P3 0.0 5.0
...etc.
```
Parameter files can also be comma-delimited if your parameter names or group names contain spaces. This should be detected automatically.

### Command-line interface

**Generate samples** (the `-p` flag is the parameter file)
```
python -m SALib.sample.saltelli \
     -n 1000 \
     -p ./SALib/test_functions/params/Ishigami.txt \
     -r model_input.txt \
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

### Parallel indices calculation (Sobol method only)
```python
Si = sobol.analyze(problem, Y,
                       calc_second_order=True, conf_level=0.95, print_to_console=False, parallel=True, n_processors=4)
```

Other methods include Morris, FAST, Delta-MIM, and DGSM. For an explanation of all command line options for each method, [see the examples here](https://github.com/SALib/SALib/tree/master/examples).


### Groups of variables (Sobol and Morris methods only)
It is sometimes useful to perform sensitivity analysis on groups of input variables to reduce the number of model runs required, when variables belong to the same component of a model, or there is some reason to believe that they should behave similarly.

Groups can be specified in two ways for the Sobol and Morris methods. First, as a fourth column in the parameter file:
```
# name lower_bound upper_bound group_name
P1 0.0 1.0 Group_1
P2 0.0 5.0 Group_2
P3 0.0 5.0 Group_2
...etc.
```

Or in the `problem` dictionary:
```python
problem = {
  'groups': ['Group_1', 'Group_2', 'Group_2'],
  'names': ['x1', 'x2', 'x3'],
  'num_vars': 3,
  'bounds': [[-3.14, 3.14], [-3.14, 3.14], [-3.14, 3.14]]
}
```

`groups` is a list of strings specifying the group name to which each variable belongs. The rest of the code stays the same:

```python
param_values = saltelli.sample(problem, 1000)
Y = Ishigami.evaluate(param_values)
Si = sobol.analyze(problem, Y, print_to_console=True)
```

But the output is printed by group:
```
Group S1 S1_conf ST ST_conf
Group_1 0.307834 0.066424 0.559577 0.082978
Group_2 0.444052 0.080255 0.667258 0.060871

Group_1 Group_2 S2 S2_conf
Group_1 Group_2 0.242964 0.124229
```
