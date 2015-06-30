'''
Created on 30 Jun 2015

@author: will2
'''

from scipy.linalg import hadamard
import numpy as np
from . import common_args
from ..util import scale_samples, read_param_file


def find_smallest(num_vars):
    for x in range(10):
        if num_vars <= 2 ** x:
            return x


def sample(problem):
    
    num_vars = problem['num_vars']
    
    # Find the smallest n, such that num_vars < k
    k = [2 ** n for n in range(16)]
    k_chosen = 2 ** find_smallest(num_vars)
    
    num_dummy_variables = k_chosen - num_vars

    # Generate the sample
    sample = np.vstack([hadamard(k_chosen), -hadamard(k_chosen)])
    
    sample = np.array((sample + 1.) / 2, dtype=np.float)
    
    bounds = list(problem['bounds'])
    if num_dummy_variables > 0:
        bounds.extend([[0, 1] for x in range(num_dummy_variables)])
    
    scale_samples(sample, bounds)
    
    return sample
    

if __name__ == "__main__":

    parser = common_args.create()
    args = parser.parse_args()

    problem = read_param_file(args.paramfile)
    param_values = sample(problem)

    np.savetxt(args.output, param_values, delimiter=args.delimiter,
               fmt='%.' + str(args.precision) + 'e')
