import numpy as np

 # Generate N x D matrix of standard normal samples
def sample(N, D):
    temp = np.random.standard_normal([N*D])
    return np.reshape(temp, (N, D))