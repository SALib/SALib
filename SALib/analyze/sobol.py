from __future__ import division
from __future__ import print_function

from scipy.stats import norm

import numpy as np

from . import common_args
from ..util import read_param_file

from multiprocessing import Pool, cpu_count
from functools import partial

EMPTY = -1

# Perform Sobol Analysis on file of model results
# Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf'
# Where each entry is a list of size D (the number of parameters)
# Containing the indices in the same order as the parameter file
def analyze(problem, Y, calc_second_order=True, num_resamples=100,
            conf_level=0.95, print_to_console=False, parallel=1,
            n_processors=1):
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

    if calc_second_order:
        S['S2'] = np.empty((D, D))
        S['S2'][:] = np.nan
        S['S2_conf'] = np.empty((D, D))
        S['S2_conf'][:] = np.nan

    r = np.random.randint(N, size=(N, num_resamples))
    Z = norm.ppf(0.5 + conf_level / 2)

    # Define number of procesors to be used if parallel option is selected
    if parallel > 0 and n_processors == 1:
        n_processors = min(
            cpu_count() if parallel == 2 else int(cpu_count() / 2 + 0.5),
            D * len(keys))

    # Create list with one entry (key, parameter 1, parameter 2) per sobol
    # index (+conf.).
    entries = [[d, j, EMPTY] for j in range(D) for d in keys]
    pool_entries = list(np.array_split(entries, n_processors))

    # Add second order (+conf.) to entries
    if calc_second_order:
        entries_second_order = [[d, j, k] for j in range(D) for k in
                                  range(j + 1, D) for d in ('S2', 'S2_conf')]
        # Split list of entries into one sublist per processor to be used
        pool_entries_second_order = list(
            np.array_split(entries_second_order, n_processors))

        # Include second order terms in the list of entries. They are spread
        # across the processes for better efficiency, as second order terms are
        # more computationaly intensive.
        pool_entries_dummy = []
        for p1o, p2o in zip(reversed(pool_entries),
                            pool_entries_second_order):
            pool_entries_dummy.append(list(p1o) + list(p2o))

        pool_entries = pool_entries_dummy

    # Calculate Sobol indexes
    if n_processors > 1:  # Parallelized calculation
        func = partial(sobol_indexes_calculation, Z, A, AB, BA, B, r)
        pool = Pool(n_processors)
        sobol_indexes_raw = pool.map(func, pool_entries)
    else:  # Serial calculation
        sobol_indexes_raw = []
        for S_entry in pool_entries:
            sobol_indexes_raw.append(
                sobol_indexes_calculation(Z, A, AB, BA, B, r, S_entry))

    # Populate results dictionary
    sobol_indexes_list = []
    for sir in sobol_indexes_raw:
        sobol_indexes_list += sir

    for si in sobol_indexes_list:  # First order (+conf.)
        if si[2] == str(EMPTY):
            S[si[0]][int(si[1])] = si[3]

    if calc_second_order:  # Second order (+conf.)
        for si in sobol_indexes_list:
            if si[2] != str(EMPTY):
                S[si[0]][int(si[1]), int(si[2])] = si[3]

    # Print results to console
    if print_to_console:

        # First order (+conf.)
        print("Parameter %s %s %s %s" % keys)

        for j in range(D):
            print("%s %f %f %f %f" % (problem['names'][j], S['S1'][
                j], S['S1_conf'][j], S['ST'][j], S['ST_conf'][j]))

        # Second order (+conf.)
        if calc_second_order:
            if print_to_console:
                print("\nParameter_1 Parameter_2 S2 S2_conf")

            for j in range(D):
                for k in range(j + 1, D):
                    if print_to_console:
                        print("%s %s %f %f" % (problem['names'][j], problem[
                            'names'][k], S['S2'][j, k], S['S2_conf'][j, k]))

    return S


def sobol_indexes_calculation(Z, A, AB, BA, B, r, entries):
    sobol_indexes = []
    for S_entry in entries:
        d = S_entry[0]
        j = S_entry[1]
        k = S_entry[2]

        if d == 'S1':
            sobol_indexes.append([d, j, k, first_order(A, AB[:, j], B)])
        elif d == 'S1_conf':
            sobol_indexes.append(
                [d, j, k, Z * first_order(A[r], AB[r, j], B[r]).std(ddof=1)])
        elif d == 'ST':
            sobol_indexes.append([d, j, k, total_order(A, AB[:, j], B)])
        elif d == 'ST_conf':
            sobol_indexes.append(
                [d, j, k, Z * total_order(A[r], AB[r, j], B[r]).std(ddof=1)])
        if d == 'S2':
            sobol_indexes.append(
                [d, j, k, second_order(A, AB[:, j], AB[:, k], BA[:, j], B)])
        elif d == 'S2_conf':
            sobol_indexes.append([d, j, k,
                                  second_order(A[r], AB[r, j], AB[r, k],
                                               BA[r, j], B[r]).std(ddof=1)])

    return sobol_indexes


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
                        choices=[1, 2],
                        help='Maximum order of sensitivity indices to '
                             'calculate')
    parser.add_argument('-r', '--resamples', type=int, required=False,
                        default=1000,
                        help='Number of bootstrap resamples for Sobol '
                             'confidence intervals')
    args = parser.parse_args()

    problem = read_param_file(args.paramfile)
    Y = np.loadtxt(args.model_output_file, delimiter=args.delimiter,
                   usecols=(args.column,))

    analyze(problem, Y, (args.max_order == 2),
            num_resamples=args.resamples, print_to_console=True)
