from __future__ import division
import numpy as np

num_vars = 8
lower_bounds = [0, 0, 0, 0, 0, 0, 0, 0]
upper_bounds = [1, 1, 1, 1, 1, 1, 1, 1]

def evaluate(values):
    a = [0, 1, 4.5, 9, 99, 99, 99, 99]
    Y = np.empty([values.shape[0]])
    
    for i, row in enumerate(values):
        Y[i] = 1.0
        
        for j in range(8):
            x = row[j]
            Y[i] *= (abs(4*x - 2) + a[j]) / (1 + a[j])
        
    return Y