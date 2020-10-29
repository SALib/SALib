import numpy as np


def evaluate(A, x, M):
    a1, a2, a3 = A.T

    return a1*x + a2*np.sin(x) + a3*np.cos(x) + x.T*(M*x)