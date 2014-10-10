from __future__ import division
import numpy as np
from . import sobol_sequence
from ..util import scale_samples, read_param_file
from . import common_args
    
# Generate matrix of Saltelli samples
# Size N x (2D + 2) if calc_second_order is True (default)
# Size N x (D + 2) otherwise
def sample(N, param_file, calc_second_order = True):

    pf = read_param_file(param_file)
    D = pf['num_vars']
    
    # How many values of the Sobol sequence to skip
    skip_values = 1000
    
    # Create base sequence - could be any type of sampling
    base_sequence = sobol_sequence.sample(N + skip_values, 2*D)
    
    if calc_second_order:
        saltelli_sequence = np.empty([(2*D + 2)*N, D])
    else:
        saltelli_sequence = np.empty([(D + 2)*N, D])
    index = 0
    
    for i in range(skip_values, N + skip_values):
        
        # Copy matrix "A"
        for j in range(D):
            saltelli_sequence[index,j] = base_sequence[i,j]
        
        index += 1
        
        # Cross-sample elements of "B" into "A"
        for k in range (D):
            for j in range(D):
                if j == k:
                    saltelli_sequence[index,j] = base_sequence[i,j+D]
                else:
                    saltelli_sequence[index,j] = base_sequence[i,j]
                
            index += 1
        
        # Cross-sample elements of "A" into "B"
        # Only needed if you're doing second-order indices (true by default)
        if calc_second_order:
            for k in range(D):
                for j in range(D):
                    if j == k:
                        saltelli_sequence[index,j] = base_sequence[i,j]
                    else:
                        saltelli_sequence[index,j] = base_sequence[i,j+D]
                
                index += 1
        
        # Copy matrix "B"
        for j in range(D):        
            saltelli_sequence[index,j] = base_sequence[i,j+D]
        
        index += 1
    
    scale_samples(saltelli_sequence, pf['bounds'])
    return saltelli_sequence

if __name__ == "__main__":

    parser = common_args.create()
    parser.add_argument('--max-order', type=int, required=False, default=2, choices=[1, 2], help='Maximum order of sensitivity indices to calculate')
    args = parser.parse_args()

    np.random.seed(args.seed)
    param_values = sample(args.samples, args.paramfile, calc_second_order=(args.max_order == 2))
    np.savetxt(args.output, param_values, delimiter=args.delimiter, fmt='%.' + str(args.precision) + 'e')
