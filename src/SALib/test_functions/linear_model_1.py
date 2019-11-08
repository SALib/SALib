import numpy as np


# Linear model that is used in Li's paper
# y = x1 + x2 + x3 + x4 + x5
def evaluate(values):
    Y = np.zeros([values.shape[0]])
    Y = np.sum(values,axis=1)

    return Y
