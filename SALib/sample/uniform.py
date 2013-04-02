import numpy as np

# Generate N x D matrix of uniform [0, 1] samples
def sample(N, D):
    return np.random.random([N, D])