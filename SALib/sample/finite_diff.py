from __future__ import division
import numpy as np
import sobol_sequence
from ..util import scale_samples, read_param_file
import common_args
    
# Generate matrix of samples for derivative-based global sensitivity measure (dgsm)
# start from a QMC (sobol) sequence and finite difference with delta % steps
def sample(N, param_file, delta = 0.01):

    pf = read_param_file(param_file)
    D = pf['num_vars']

    # How many values of the Sobol sequence to skip
    skip_values = 1000
    
    # Create base sequence - could be any type of sampling
    base_sequence = sobol_sequence.sample(N + skip_values, D)
    scale_samples(base_sequence, pf['bounds']) # scale before finite differencing
    dgsm_sequence = np.empty([N*(D+1), D])
    
    index = 0
    
    for i in xrange(skip_values, N + skip_values):
        
        # Copy the initial point
        dgsm_sequence[index,:] = base_sequence[i,:]
        index += 1

        for j in xrange(D):
            temp = np.zeros(D)
            temp[j] = base_sequence[i,j]*delta
            dgsm_sequence[index,:] = base_sequence[i,:] + temp
            dgsm_sequence[index,j] = min(dgsm_sequence[index,j], pf['bounds'][j][1])
            dgsm_sequence[index,j] = max(dgsm_sequence[index,j], pf['bounds'][j][0])
            index += 1
            
    return dgsm_sequence

if __name__ == "__main__":

    parser = common_args.create()
    parser.add_argument('-d','--delta', type=float, required=False, default=0.01, help='Finite difference step size (percent)')
    args = parser.parse_args()

    np.random.seed(args.seed)
    param_values = sample(args.samples, args.paramfile, args.delta)
    np.savetxt(args.output, param_values, delimiter=args.delimiter, fmt='%.' + str(args.precision) + 'e')
