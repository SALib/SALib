from __future__ import division

import numpy as np

from . import common_args
from . import sobol_sequence
from ..util import scale_samples, read_param_file


def sample(problem, N, calc_second_order=True,groups=False):
    """Generates model inputs using Saltelli's extension of the Sobol sequence.

    Returns a NumPy matrix containing the model inputs using Saltelli's sampling
    scheme.  Saltelli's scheme extends the Sobol sequence in a way to reduce
    the error rates in the resulting sensitivity index calculations.  If
    calc_second_order is False, the resulting matrix has N * (D + 2)
    rows, where D is the number of parameters.  If calc_second_order is True,
    the resulting matrix has N * (2D + 2) rows.  These model inputs are
    intended to be used with :func:`SALib.analyze.sobol.analyze`.

    Parameters
    ----------
    problem : dict
        The problem definition
    N : int
        The number of samples to generate
    calc_second_order : bool
        Calculate second-order sensitivities (default True)
    groups : bool
        Cross-sample based on groups of parameters
    """
    # finding number of groups of groups, Dg
    # if groups = False, number of groups is the number of variables
    if groups:
        Dg = len(problem['groups'])
    else:
        Dg = problem['num_vars']

    # number of variables
    D = problem['num_vars']

    # How many values of the Sobol sequence to skip
    skip_values = 1000

    # Create base sequence - could be any type of sampling
    base_sequence = sobol_sequence.sample(N + skip_values, 2 * D)

    if calc_second_order:
        saltelli_sequence = np.empty([(2 * Dg + 2) * N, D])
    else:
        saltelli_sequence = np.empty([(Dg + 2) * N, D])
    index = 0

    for i in range(skip_values, N + skip_values):

        # Copy matrix "A"
        for j in range(D):
            saltelli_sequence[index, j] = base_sequence[i, j]

        index += 1

        # Cross-sample elements of "B" into "A"
        if groups:
            # method of cross-sampling "B" into "A" for groups
            # groups that are "off-diagonal" (l != m) will be drawn from "A"
            # groups that are "on-diagonal" (l = m) will be drawn from "B"
            for l in range(len(problem['groups'])):
                for m in range(len(problem['groups'])):
                    if l == m:
                        # condition for being "on-diagonal" group
                        # "on-diagonal" groups have elements taken from "B"
                        for k in range(len(problem['groups'][m])):
                            j = problem['names'].index(problem['groups'][m][k])
                            saltelli_sequence[index, j] = base_sequence[i, j + D]
                    else:
                        # condition for being "off-diagonal" group
                        # "off-diagonal" groups have elements taken from "A"
                        for k in range(len(problem['groups'][m])):
                            j = problem['names'].index(problem['groups'][m][k])
                            saltelli_sequence[index, j] = base_sequence[i, j]

                index += 1
        else:
            # cross-sampling without groups
            for k in range(D):
                for j in range(D):
                    if j == k:
                        saltelli_sequence[index, j] = base_sequence[i, j + D]
                    else:
                        saltelli_sequence[index, j] = base_sequence[i, j]

                index += 1

        # Cross-sample elements of "A" into "B"
        # Only needed if you're doing second-order indices (true by default)
        if calc_second_order:
            if groups:
                # method of cross-sampling "A" into "B" for groups
                # groups that are "off-diagonal" (l != m) will be drawn from "B"
                # groups that are "on-diagonal" (l = m) will be drawn from "A"
                for l in range(len(problem['groups'])):
                    for m in range(len(problem['groups'])):
                        if l == m:
                            # condition for being "on-diagonal" group
                            # "on-diagonal" groups have elements taken from "A"
                            for k in range(len(problem['groups'][m])):
                                j = problem['names'].index(problem['groups'][m][k])
                                saltelli_sequence[index, j] = base_sequence[i, j]
                        else:
                            # condition for being "off-diagonal" group
                            # "off-diagonal" groups have elements taken from "B"
                            for k in range(len(problem['groups'][m])):
                                j = problem['names'].index(problem['groups'][m][k])
                                saltelli_sequence[index, j] = base_sequence[i, j+D]

                    index += 1

            else:
                # cross-sampling without groups
                for k in range(D):
                    for j in range(D):
                        if j == k:
                            saltelli_sequence[index, j] = base_sequence[i, j]
                        else:
                            saltelli_sequence[index, j] = base_sequence[i, j + D]

                    index += 1

        # Copy matrix "B"
        for j in range(D):
            saltelli_sequence[index, j] = base_sequence[i, j + D]

        index += 1

    scale_samples(saltelli_sequence, problem['bounds'])
    return saltelli_sequence

if __name__ == "__main__":

    parser = common_args.create()

    parser.add_argument(
        '-n', '--samples', type=int, required=True, help='Number of Samples')

    parser.add_argument('--max-order', type=int, required=False, default=2,
                        choices=[1, 2], help='Maximum order of sensitivity indices to calculate')
    args = parser.parse_args()

    np.random.seed(args.seed)
    problem = read_param_file(args.paramfile)

    param_values = sample(problem, args.samples, calc_second_order=(args.max_order == 2))
    np.savetxt(args.output, param_values, delimiter=args.delimiter,
               fmt='%.' + str(args.precision) + 'e')
