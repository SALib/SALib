from __future__ import division
from __future__ import print_function

from scipy.stats import norm

import numpy as np
import random as rd

from . import common_args
from ..util import read_param_file

from multiprocessing import Pool, cpu_count
from functools import partial
try:
    from itertools import zip_longest
except ImportError:
    # Python 2
    from itertools import izip_longest as zip_longest


# Perform Sobol Analysis on a vector of model results
# Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf'
# Where each entry is a list of size D (the number of parameters)
# Containing the indices in the same order as the parameter file
def analyze(problem, Y, calc_second_order=True, num_resamples=100,
            conf_level=0.95, print_to_console=False, parallel=False,
            n_processors=None):

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

    A,B,AB,BA = separate_output_values(Y, D, N, calc_second_order)
    r = np.random.randint(N, size=(N, num_resamples))
    Z = norm.ppf(0.5 + conf_level / 2)

    if not parallel:
        S = create_Si_dict(D, calc_second_order)

        for j in range(D):
            S['S1'][j] = first_order(A, AB[:, j], B)
            S['S1_conf'][j] = Z * first_order(A[r], AB[r,j], B[r]).std(ddof=1)
            S['ST'][j] = total_order(A, AB[:, j], B)
            S['ST_conf'][j] = Z * total_order(A[r], AB[r,j], B[r]).std(ddof=1)

        # Second order (+conf.)
        if calc_second_order:
            for j in range(D):
                for k in range(j + 1, D):
                    S['S2'][j, k] = second_order(
                        A, AB[:, j], AB[:, k], BA[:, j], B)
                    S['S2_conf'][j, k] = Z * second_order(A[r], AB[r, j], 
                        AB[r, k], BA[r, j], B[r]).std(ddof=1)

    else:            
        tasks = create_task_list(D, calc_second_order, n_processors)

        func = partial(sobol_parallel, Z, A, AB, BA, B, r)
        pool = Pool(n_processors)
        S_list = pool.map_async(func, tasks)
        pool.close()
        pool.join()

        S = Si_list_to_dict(S_list.get(), D, calc_second_order)

    # Print results to console
    if print_to_console:
        print_indices(S, problem, calc_second_order)


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


def create_Si_dict(D, calc_second_order):
    # initialize empty dict to store sensitivity indices
    S = dict((k, np.empty(D)) for k in ('S1', 'S1_conf', 'ST', 'ST_conf'))

    if calc_second_order:
        S['S2'] = np.empty((D, D))
        S['S2'][:] = np.nan
        S['S2_conf'] = np.empty((D, D))
        S['S2_conf'][:] = np.nan

    return S


def separate_output_values(Y, D, N, calc_second_order): 
    AB = np.empty((N, D))
    BA = np.empty((N, D)) if calc_second_order else None
    step = 2 * D + 2 if calc_second_order else D + 2

    A = Y[0:Y.size:step]
    B = Y[(step - 1):Y.size:step]
    for j in range(D):
        AB[:, j] = Y[(j + 1):Y.size:step]
        if calc_second_order:
            BA[:, j] = Y[(j + 1 + D):Y.size:step]

    return A,B,AB,BA


def sobol_parallel(Z, A, AB, BA, B, r, tasks):
    sobol_indices = []
    for d,j,k in tasks:
        if d == 'S1':
            s = first_order(A, AB[:, j], B)
        elif d == 'S1_conf':
            s = Z * first_order(A[r], AB[r, j], B[r]).std(ddof=1)
        elif d == 'ST':
            s = total_order(A, AB[:, j], B)
        elif d == 'ST_conf':
            s = Z * total_order(A[r], AB[r, j], B[r]).std(ddof=1)
        if d == 'S2':
            s = second_order(A, AB[:, j], AB[:, k], BA[:, j], B)
        elif d == 'S2_conf':
            s = Z * second_order(A[r], AB[r, j], AB[r, k],
                                               BA[r, j], B[r]).std(ddof=1)
        sobol_indices.append([d, j, k, s])

    return sobol_indices


def create_task_list(D, calc_second_order, n_processors):
    # Create list with one entry (key, parameter 1, parameter 2) per sobol
    # index (+conf.). This is used to supply parallel tasks to multiprocessing.Pool
    tasks_first_order = [[d, j, None] for j in range(D) for d in ('S1', 'S1_conf', 'ST', 'ST_conf')]
    
    # Add second order (+conf.) to tasks
    tasks_second_order = []
    if calc_second_order:
        tasks_second_order = [[d, j, k] for j in range(D) for k in
                            range(j + 1, D) for d in ('S2', 'S2_conf')]

    if n_processors == None:
        n_processors = min(cpu_count(), len(tasks_first_order) + len(tasks_second_order))

    if not calc_second_order:
        tasks = np.array_split(tasks_first_order, n_processors)
    else:
        # merges both lists alternating its elements and splits the resulting list into n_processors sublists
        tasks = np.array_split([v for v in sum(
            zip_longest(tasks_first_order[::-1], tasks_second_order), ()) 
                if v is not None], n_processors)

    return tasks


def Si_list_to_dict(S_list, D, calc_second_order):
    # Convert the parallel output into the regular dict format for printing/returning
    S = create_Si_dict(D, calc_second_order)
    L = []
    for l in S_list: # first reformat to flatten
        L += l

    for s in L:  # First order (+conf.)
        if s[2] is None:
            S[s[0]][s[1]] = s[3]
        else:
            S[s[0]][s[1], s[2]] = s[3]

    return S


def print_indices(S, problem, calc_second_order):
    # Output to console
    D = problem['num_vars']
    print('Parameter S1 S1_conf ST ST_conf')

    for j in range(D):
        print('%s %f %f %f %f' % (problem['names'][j], S['S1'][
            j], S['S1_conf'][j], S['ST'][j], S['ST_conf'][j]))

    if calc_second_order:
        print('\nParameter_1 Parameter_2 S2 S2_conf')

        for j in range(D):
            for k in range(j + 1, D):
                print("%s %s %f %f" % (problem['names'][j], problem[
                    'names'][k], S['S2'][j, k], S['S2_conf'][j, k]))


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
