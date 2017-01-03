'''
Created on 30 Jun 2015

@author: will2
'''

from __future__ import print_function
import numpy as np
from . import common_args
from SALib.util import read_param_file, unscale_samples
from SALib.sample.ff import generate_contrast, sample, extend_bounds

def analyze(problem, X, Y, second_order=False, print_to_console=False):
    """Perform a fractional factorial analysis

    Returns a dictionary with keys 'ME' (main effect) and 'IE' (interaction
    effect). The techniques bulks out the number of parameters with dummy
    parameters to the nearest 2**n.  Any results involving dummy parameters
    could indicate a problem with the model runs.

    Arguments
    ---------
    problem: dict
        The problem definition
    X: numpy.matrix
        The NumPy matrix containing the model inputs
    Y: numpy.array
        The NumPy array containing the model outputs
    second_order: bool, default=False
        Include interaction effects
    print_to_console: bool, default=False
        Print results directly to console

    Returns
    -------
    Si: dict
        A dictionary of sensitivity indices, including main effects ``ME``,
        and interaction effects ``IE`` (if ``second_order`` is True)

    Examples
    --------
    >>> X = sample(problem)
    >>> Y = X[:, 0] + (0.1 * X[:, 1]) + ((1.2 * X[:, 2]) * (0.2 + X[:, 0]))
    >>> analyze(problem, X, Y, second_order=True, print_to_console=True)
    """

    problem = extend_bounds(problem)
    num_vars = problem['num_vars']

    X = generate_contrast(problem)

    main_effect = (1. / (2 * num_vars)) * np.dot(Y, X)

    Si = dict((k, [None] * num_vars)
              for k in ['names', 'ME'])
    Si['ME'] = main_effect
    Si['names'] = problem['names']

    if print_to_console:
        print("Parameter ME")
        for j in range(num_vars):
            print("%s %f" % (problem['names'][j], Si['ME'][j]))

    if second_order == True:
        interaction_names, interaction_effects = interactions(problem,
                                                              Y,
                                                              print_to_console)

        Si['names'].append(interaction_names)
        Si['IE'] = interaction_effects

    return Si


def interactions(problem, Y, print_to_console=False):
    """Computes the second order effects

    Computes the second order effects (interactions) between
    all combinations of pairs of input factors

    Arguments
    ---------
    problem: dict
        The problem definition
    Y: numpy.array
        The NumPy array containing the model outputs
    print_to_console: bool, default=False
        Print results directly to console

    Returns
    -------
    ie_names: list
        The names of the interaction pairs
    IE: list
        The sensitivity indices for the pairwise interactions

    """

    names = problem['names']
    num_vars = problem['num_vars']

    X = generate_contrast(problem)

    ie_names = []
    IE = []

    for col in range(X.shape[1]):
        for col_2 in range(col):
            x = X[:, col] * X[:, col_2]
            var_names = names[col_2] + names[col]
            ie_names.append(var_names)
            IE.append((1. / (2 * num_vars)) * np.dot(Y, x))
    if print_to_console:
        [print('%s %f' % (n, i) ) for (n, i) in zip(ie_names, IE) ]

    return ie_names, IE

if __name__ == "__main__":

    parser = common_args.create()
    parser.add_argument('-X', '--model-input-file', type=str,
                        required=True, default=None, help='Model input file')
    parser.add_argument('--max-order', type=int, required=False, default=2,
                    choices=[1, 2], help='Maximum order of sensitivity indices to calculate')
    args = parser.parse_args()

    problem = read_param_file(args.paramfile)

    Y = np.loadtxt(args.model_output_file, delimiter=args.delimiter, usecols=(args.column,))
    X = np.loadtxt(args.model_input_file, delimiter=args.delimiter, ndmin=2)
    if len(X.shape) == 1:
        X = X.reshape((len(X), 1))
    analyze(problem, X, Y, (args.max_order == 2), print_to_console=True)
