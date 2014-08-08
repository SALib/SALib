from __future__ import division
import numpy as np

# Generate N x D matrix of latin hypercube samples
def sample(N, D):
    
    result = np.empty([N, D])
    temp = np.empty([N])
    d = 1.0 / N
    
    for i in range(D):
        
        for j in range(N):
            temp[j] = np.random.uniform(low = j*d, high = (j+1)*d, size = 1)[0]
        
        np.random.shuffle(temp)
        
        for j in range(N):
            result[j,i] = temp[j]
    
    return result