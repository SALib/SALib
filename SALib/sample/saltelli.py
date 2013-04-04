from __future__ import division
import numpy as np
from . import sobol
    
# Generate matrix of Saltelli samples
# Size N x (2D + 2) if calc_second_order is True (default)
# Size N x (D + 2) otherwise
def sample(N, D, calc_second_order = True):
    
    # How many values of the Sobol sequence to skip
    skip_values = 1000
    
    # Create base sequence - could be any type of sampling
    base_sequence = sobol.sample(N + skip_values, 2*D)
    
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
    
    return saltelli_sequence