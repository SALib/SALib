'''
Created on 30 Jun 2015

@author: will2
'''

import numpy as np
from . import common_args
from ..util import read_param_file

def analyze(problem, X, Y, second_order=False, print_to_console=False):
    
    num_vars = problem['num_vars']
    number_of_vars_in_input_sample = X.shape[1]
    
    number_of_dummy_vars = number_of_vars_in_input_sample - num_vars
    
    names = problem['names']
    names.extend(["dummy_" + str(var) for var in range(number_of_dummy_vars)])
    
    main_effect = (1. / (2 * number_of_vars_in_input_sample)) * np.dot(Y, X)
    
    Si = dict((k, [None] * num_vars)
              for k in ['names', 'ME'])
    Si['ME'] = main_effect
    Si['names'] = names
        
    if print_to_console:
        print("Parameter ME")
        for j in range(num_vars):
            print("%s %f" % (problem['names'][j], Si['ME'][j]))
    
    if second_order == True:
        interactions(problem, X, Y, print_to_console)
    
    return Si

def interactions(problem, X, Y, print_to_console=False):
    
    names = problem['names']
    
    for col in range(X.shape[1]):
        for col_2 in range(col):
            x = X[:, col] * X[:, col_2]
            if print_to_console:
                var_names = names[col_2] + names[col]
                print ('%s %f' % (var_names, (1. / (2 * problem['num_vars'])) * np.dot(Y, x)))

if __name__ == "__main__":

    parser = common_args.create()
    parser.add_argument('-X', '--model-input-file', type=str,
                        required=True, default=None, help='Model input file')
    args = parser.parse_args()

    problem = read_param_file(args.paramfile)

    Y = np.loadtxt(args.model_output_file, delimiter=args.delimiter, usecols=(args.column,))
    X = np.loadtxt(args.model_input_file, delimiter=args.delimiter, ndmin=2)
    if len(X.shape) == 1:
        X = X.reshape((len(X), 1))

    analyze(problem, X, Y, num_resamples=args.resamples, print_to_console=True,
            num_levels=args.levels, grid_jump=args.grid_jump)
