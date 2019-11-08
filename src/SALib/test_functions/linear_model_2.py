import numpy as np


# Linear model that is used in Li's paper
# y = 5x1 + 4x2 + 3x3 + 2x4 + x5
def evaluate(values):
    Y = np.zeros([values.shape[0]])
    Y = 5 * values[:,0] + 4 * values[:,1] + 3 * values[:,2] + 2 * values[:,3] + values[:,4] 

    return Y
