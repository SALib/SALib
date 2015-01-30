from __future__ import division
import numpy as np
from . import common_args
from ..util import scale_samples, read_param_file

# Generate N x D matrix of latin hypercube samples


def sample(problem, N):

    D = problem['num_vars']

    result = np.empty([N, D])
    temp = np.empty([N])
    d = 1.0 / N

    for i in range(D):

        for j in range(N):
            temp[j] = np.random.uniform(
                low=j * d, high=(j + 1) * d, size = 1)[0]

        np.random.shuffle(temp)

        for j in range(N):
            result[j, i] = temp[j]

    scale_samples(result, problem['bounds'])
    return result

if __name__ == "__main__":

    parser = common_args.create()
    args = parser.parse_args()

    np.random.seed(args.seed)
    problem = read_param_file(args.paramfile)

    param_values = sample(problem, args.samples)
    np.savetxt(args.output, param_values, delimiter=args.delimiter,
               fmt='%.' + str(args.precision) + 'e')
