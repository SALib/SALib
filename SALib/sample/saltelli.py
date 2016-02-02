from __future__ import division

import numpy as np

from . import common_args
from . import sobol_sequence
from ..util import scale_samples, nonuniform_scale_samples, read_param_file


def sample(problem, N, calc_second_order=True):
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
    """
    D = problem['num_vars']

    if not problem.get('groups'):
        groups = False
        Dg = problem['num_vars']
    else:
        groups = True
        # condition for when problem was defined from parameter file
        # can access the 'groups' tuple (matrix, list of unique group names)
        # to determine the number of groups
        # if problem defined as a dictionary in the code, find the number
        # of unique group names
        # also make matrix to account for group names
        if len(problem['groups']) == 2:
            Dg = len(problem['groups'][1])
        else:
            Dg = len(np.unique(problem['groups']))
            gp_mat = np.zeros([D, Dg])
            for i in range(Dg):
                # group name to check for equivalency
                groupNameIt = np.unique(problem['groups'])[i]
                for j in range(D):
                    if problem['groups'][j] == groupNameIt:
                        gp_mat[j,i] = 1
            # making a tuple similar to the one made by the read_param_file
            # for use later in the code
            problem['groups'] = (gp_mat,np.unique(problem['groups']))

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
        # condition for group sampling (groups is True)
        if groups:
            # method of cross-sampling "B" into "A" for groups
            # groups that are "off-diagional" (l != m) will be form "A"
            # groups that are "on-diagional" (l = m) will be from "B"
            for l in range(Dg):
                for m in range(D):
                    if problem['groups'][0][m,l] == 1:
                        saltelli_sequence[index, m] = base_sequence[i, m + D]
                    else:
                        saltelli_sequence[index, m] = base_sequence[i, m]

                index += 1
        else:
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
            # condition for group sampling (groups is True)
            if groups:
                # method of cross-sampling "A" into "B" for groups
                # groups that are "off-diagional" (l != m) will be form "B"
                # groups that are "on-diagional" (l = m) will be from "A"
                for l in range(Dg):
                    for m in range(D):
                        if problem['groups'][0][m,l] == 1:
                            saltelli_sequence[index, m] = base_sequence[i, m]
                        else:
                            saltelli_sequence[index, m] = base_sequence[i, m + D]

                    index += 1
            else:
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
    if not problem.get('dists'):
        # scaling values out of 0-1 range with uniform distributions
        scale_samples(saltelli_sequence,problem['bounds'])
        return saltelli_sequence
    else:
        # scaling values to other distributions based on inverse CDFs
        scaled_saltelli = nonuniform_scale_samples(saltelli_sequence,problem['bounds'],problem['dists'])
        return scaled_saltelli

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
