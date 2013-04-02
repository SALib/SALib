from __future__ import division
import numpy as np
import math
    
# Generate N x D matrix of extended FAST samples (Saltelli 1999)
def sample(N, D, M = 4):
    
    omega = np.empty([D])
    omega[0] = math.floor((N - 1) / (2 * M))
    m = math.floor(omega[0] / (2 * M))
    
    if m >= (D-1):
        omega[1:] = np.floor(np.linspace(1, m, D-1)) 
    else:
        omega[1:] = np.arange(D-1) % m + 1

    # Discretization of the frequency space, s
    s = (2 * math.pi / N) * np.arange(N)
    
    # Transformation to get points in the X space
    X = np.empty([N*D, D])
    omega2 = np.empty([D])
    
    for i in range(D):
        omega2[i] = omega[0]
        idx = range(i) + range(i+1,D)
        omega2[idx] = omega[1:]
        l = range(i*N, (i+1)*N)
        
        for j in range(D):
            g = 0.5 + (1/math.pi) * np.arcsin(np.sin(omega2[j] * s))
            X[l,j] = g
        
    return X