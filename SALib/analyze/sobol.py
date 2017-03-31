from __future__ import division
from __future__ import print_function

from scipy.stats import norm

import numpy as np

from . import common_args
from ..util import read_param_file, compute_groups_matrix

from multiprocessing import Pool, cpu_count
from functools import partial
try:
    from itertools import zip_longest
except ImportError:
    # Python 2
    from itertools import izip_longest as zip_longest


def analyze(problem, Y, calc_second_order=True, num_resamples=100,
            conf_level=0.95, print_to_console=False, parallel=False,
            n_processors=None):
    """Perform Sobol Analysis on model outputs.

    Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf', where
    each entry is a list of size D (the number of parameters) containing the
    indices in the same order as the parameter file.  If calc_second_order is
    True, the dictionary also contains keys 'S2' and 'S2_conf'.

    Parameters
    ----------
    problem : dict
        The problem definition
    Y : numpy.array
        A NumPy array containing the model outputs
    calc_second_order : bool
        Calculate second-order sensitivities (default True)
    num_resamples : int
        The number of resamples (default 100)
    conf_level : float
        The confidence interval level (default 0.95)
    print_to_console : bool
        Print results directly to console (default False)

    References
    ----------
    .. [1] Sobol, I. M. (2001).  "Global sensitivity indices for nonlinear
           mathematical models and their Monte Carlo estimates."  Mathematics
           and Computers in Simulation, 55(1-3):271-280,
           doi:10.1016/S0378-4754(00)00270-6.
    .. [2] Saltelli, A. (2002).  "Making best use of model evaluations to
           compute sensitivity indices."  Computer Physics Communications,
           145(2):280-297, doi:10.1016/S0010-4655(02)00280-1.
    .. [3] Saltelli, A., P. Annoni, I. Azzini, F. Campolongo, M. Ratto, and
           S. Tarantola (2010).  "Variance based sensitivity analysis of model
           output.  Design and estimator for the total sensitivity index."
           Computer Physics Communications, 181(2):259-270,
           doi:10.1016/j.cpc.2009.09.018.

    Examples
    --------
    >>> X = saltelli.sample(problem, 1000)
    >>> Y = Ishigami.evaluate(X)
    >>> Si = sobol.analyze(problem, Y, print_to_console=True)
    
    """
    # determining if groups are defined and adjusting the number
    # of rows in the cross-sampled matrix accordingly
    if not problem.get('groups'):
        D = problem['num_vars']
    else:
        D = len(set(problem['groups']))

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

    # normalize the model output
    Y = (Y - Y.mean())/Y.std()
    
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
        tasks, n_processors = create_task_list(D, calc_second_order, n_processors)

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
    S = dict((k, np.zeros(D)) for k in ('S1', 'S1_conf', 'ST', 'ST_conf'))

    if calc_second_order:
        S['S2'] = np.zeros((D, D))
        S['S2'][:] = np.nan
        S['S2_conf'] = np.zeros((D, D))
        S['S2_conf'][:] = np.nan

    return S


def separate_output_values(Y, D, N, calc_second_order):
    AB = np.zeros((N, D))
    BA = np.zeros((N, D)) if calc_second_order else None
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
        elif d == 'S2':
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

    if n_processors is None:
        n_processors = min(cpu_count(), len(tasks_first_order) + len(tasks_second_order))

    if not calc_second_order:
        tasks = np.array_split(tasks_first_order, n_processors)
    else:
        # merges both lists alternating its elements and splits the resulting list into n_processors sublists
        tasks = np.array_split([v for v in sum(
            zip_longest(tasks_first_order[::-1], tasks_second_order), ())
                if v is not None], n_processors)

    return tasks, n_processors


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
    if not problem.get('groups'):
        title = 'Parameter'
        names = problem['names']
        D = problem['num_vars']
    else:
        title = 'Group'
        _,names = compute_groups_matrix(problem['groups'], problem['num_vars'])
        D = len(names)

    print('%s S1 S1_conf ST ST_conf' % title)

    for j in range(D):
        print('%s %f %f %f %f' % (names[j], S['S1'][
            j], S['S1_conf'][j], S['ST'][j], S['ST_conf'][j]))

    if calc_second_order:
        print('\n%s_1 %s_2 S2 S2_conf' % (title,title))

        for j in range(D):
            for k in range(j + 1, D):
                print("%s %s %f %f" % (names[j], names[k],
                    S['S2'][j, k], S['S2_conf'][j, k]))

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
    parser.add_argument('--parallel', action='store_true', help='Makes '
                        'use of parallelization.',
                        dest='parallel')
    parser.add_argument('--processors', type=int, required=False,
                        default=None,
                        help='Number of processors to be used with the ' +
                        'parallel option.', dest='n_processors')
    args = parser.parse_args()

    problem = read_param_file(args.paramfile)
    Y = np.loadtxt(args.model_output_file, delimiter=args.delimiter,
                   usecols=(args.column,))

    analyze(problem, Y, (args.max_order == 2),
            num_resamples=args.resamples, print_to_console=True,
            parallel=args.parallel, n_processors=args.n_processors)
