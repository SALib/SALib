from __future__ import division

import math

import numpy as np

from . import common_args
from .. util import scale_samples, read_param_file


# Generate N x D matrix of extended FAST samples (Saltelli 1999)
def sample(problem, N, M=4):

    D = problem['num_vars']

    omega = np.empty([D])
    omega[0] = math.floor((N - 1) / (2 * M))
    m = math.floor(omega[0] / (2 * M))

    if m >= (D - 1):
        omega[1:] = np.floor(np.linspace(1, m, D - 1))
    else:
        omega[1:] = np.arange(D - 1) % m + 1

    # Discretization of the frequency space, s
    s = (2 * math.pi / N) * np.arange(N)

    # Transformation to get points in the X space
    X = np.empty([N * D, D])
    omega2 = np.empty([D])

    for i in range(D):
        omega2[i] = omega[0]
        idx = list(range(i)) + list(range(i + 1, D))
        omega2[idx] = omega[1:]
        l = range(i * N, (i + 1) * N)

        # random phase shift on [0, 2pi) following Saltelli et al.
        # Technometrics 1999
        phi = 2 * math.pi * np.random.rand()

        for j in range(D):
            g = 0.5 + (1 / math.pi) * np.arcsin(np.sin(omega2[j] * s + phi))
            X[l, j] = g

    scale_samples(X, problem['bounds'])
    return X

if __name__ == "__main__":

    parser = common_args.create()
    parser.add_argument(
        '-n', '--samples', type=int, required=True, help='Number of Samples')
   
    parser.add_argument(
        '-M', type=int, required=False, default=4, help='M coefficient, default 4')
    args = parser.parse_args()

    np.random.seed(args.seed)
    problem = read_param_file(args.paramfile)

    param_values = sample(problem, N=args.samples, M=args.M)
    np.savetxt(args.output, param_values, delimiter=args.delimiter,
               fmt='%.' + str(args.precision) + 'e')
