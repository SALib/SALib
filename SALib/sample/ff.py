'''
Created on 30 Jun 2015

@author: will2
'''

from scipy.linalg import hadamard
import numpy as np
from . import common_args
from ..util import scale_samples, read_param_file


def find_smallest(num_vars):
    '''
    Find the smallest exponent of two that is greater than the number
    of variables
    '''
    for x in range(10):
        if num_vars <= 2 ** x:
            return x


def extend_bounds(problem):
    
    num_vars = problem['num_vars']
    num_ff_vars = 2 ** find_smallest(num_vars)
    num_dummy_variables = num_ff_vars - num_vars
    
    bounds = list(problem['bounds'])
    names = problem['names']
    if num_dummy_variables > 0:
        bounds.extend([[0, 1] for x in range(num_dummy_variables)])
        names.extend(["dummy_" + str(var) for var in range(num_dummy_variables)])
        problem['bounds'] = bounds
        problem['names'] = names
        problem['num_vars'] = num_ff_vars
    
    return problem

def generate_contrast(problem):
    
    num_vars = problem['num_vars']
    
    # Find the smallest n, such that num_vars < k
    k = [2 ** n for n in range(16)]
    k_chosen = 2 ** find_smallest(num_vars)
    
    # Generate the fractional factorial contrast
    contrast = np.vstack([hadamard(k_chosen), -hadamard(k_chosen)])
    
    return contrast
    
def sample(problem):
    
    contrast = generate_contrast(problem)
    sample = np.array((contrast + 1.) / 2, dtype=np.float)
    problem = extend_bounds(problem)
    scale_samples(sample, problem['bounds'])
    return sample

if __name__ == "__main__":

    parser = common_args.create()
    args = parser.parse_args()

    problem = read_param_file(args.paramfile)
    param_values = sample(problem)

    np.savetxt(args.output, param_values, delimiter=args.delimiter,
               fmt='%.' + str(args.precision) + 'e')
