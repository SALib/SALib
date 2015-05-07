from __future__ import division
from __future__ import print_function

from scipy.stats import norm

import numpy as np

from . import common_args
from ..util import read_param_file


# Perform Sobol Analysis on file of model results
# Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf'
# Where each entry is a list of size D (the number of parameters)
# Containing the indices in the same order as the parameter file
def analyze(problem, Y, calc_second_order=True, num_resamples=100,
            conf_level=0.95, print_to_console=False):

    D = problem['num_vars']

    if calc_second_order and Y.size % (2 * D + 2) == 0:
        N = int(Y.size / (2 * D + 2))
    elif not calc_second_order and Y.size % (D + 2) == 0:
        N = int(Y.size / (D + 2))
    else:
        raise RuntimeError("""
        Incorrect number of samples in model output file. 
        Confirm that calc_second_order matches option used during sampling.""")

    if conf_level < 0 or conf_level > 1:
        raise RuntimeError("Confidence level must be between 0-1.")

    A = np.empty(N)
    B = np.empty(N)
    AB = np.empty((N, D))
    BA = np.empty((N, D)) if calc_second_order else None
    step = 2 * D + 2 if calc_second_order else D + 2

    A = Y[0:Y.size:step]
    B = Y[(step - 1):Y.size:step]
    for j in range(D):
        AB[:, j] = Y[(j + 1):Y.size:step]
        if calc_second_order:
            BA[:, j] = Y[(j + 1 + D):Y.size:step]

    # First order (+conf.) and Total order (+conf.)
    keys = ('S1', 'S1_conf', 'ST', 'ST_conf')
    S = dict((k, np.empty(D)) for k in keys)
    if print_to_console:
        print("Parameter %s %s %s %s" % keys)

    r = np.random.randint(N, size=(N, num_resamples))
    Z = norm.ppf(0.5 + conf_level / 2)

    for j in range(D):
        S['S1'][j] = first_order(A, AB[:, j], B)
        S['S1_conf'][j] = Z * first_order(A[r], AB[r,j], B[r]).std(ddof=1)
        S['ST'][j] = total_order(A, AB[:, j], B)
        S['ST_conf'][j] = Z * total_order(A[r], AB[r,j], B[r]).std(ddof=1)

        if print_to_console:
            print("%s %f %f %f %f" % (problem['names'][j], S['S1'][
                  j], S['S1_conf'][j], S['ST'][j], S['ST_conf'][j]))

    # Second order (+conf.)
    if calc_second_order:
        S['S2'] = np.empty((D, D))
        S['S2'][:] = np.nan
        S['S2_conf'] = np.empty((D, D))
        S['S2_conf'][:] = np.nan
        if print_to_console:
            print("\nParameter_1 Parameter_2 S2 S2_conf")

        for j in range(D):
            for k in range(j + 1, D):
                S['S2'][j, k] = second_order(
                    A, AB[:, j], AB[:, k], BA[:, j], B)
                S['S2_conf'][j, k] = Z * second_order(A[r], AB[r, j], 
                    AB[r, k], BA[r, j], B[r]).std(ddof=1)

                if print_to_console:
                    print("%s %s %f %f" % (problem['names'][j], problem[
                          'names'][k], S['S2'][j, k], S['S2_conf'][j, k]))

    return S


def first_order(A, AB, B):
    # First order estimator following Saltelli et al. 2010 CPC, normalized by
    # sample variance
    return np.mean(B * (AB - A), axis=0) / np.var(np.r_[A, B], axis=0)


def total_order(A, AB, B):
    # Total order estimator following Saltelli et al. 2010 CPC, normalized by
    # sample variance
    return 0.5 * np.mean((A - AB) ** 2, axis=0) / np.var(np.r_[A, B], axis=0)


def second_order(A, ABj, ABk, BAj, B):
    # Second order estimator following Saltelli 2002
    Vjk = np.mean(BAj * ABk - A * B, axis=0) / np.var(np.r_[A, B], axis=0)
    Sj = first_order(A, ABj, B)
    Sk = first_order(A, ABk, B)

    return Vjk - Sj - Sk


if __name__ == "__main__":
    parser = common_args.create()
    parser.add_argument('--max-order', type=int, required=False, default=2,
                        choices=[1, 2], help='Maximum order of sensitivity indices to calculate')
    parser.add_argument('-r', '--resamples', type=int, required=False, default=1000,
                        help='Number of bootstrap resamples for Sobol confidence intervals')
    args = parser.parse_args()

    problem = read_param_file(args.paramfile)
    Y = np.loadtxt(args.model_output_file, delimiter=args.delimiter, usecols=(args.column,))

    analyze(problem, Y, (args.max_order == 2),
            num_resamples=args.resamples, print_to_console=True)
