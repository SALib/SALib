#!/usr/bin/env python
# coding=utf8

import numpy as np
from scipy.signal import periodogram

from . import common_args
from ..util import (read_param_file, ResultDict, extract_groups, 
                    compute_groups_matrix, _group_metric)


def analyze(problem, X, Y, M=10, print_to_console=False, seed=None):
    """Performs the Random Balanced Design - Fourier Amplitude Sensitivity Test
    (RBD-FAST) on model outputs.

    Returns a dictionary with keys 'S1', where each entry is a list of
    size D (the number of parameters) containing the indices in the same order
    as the parameter file.

    Parameters
    ----------
    problem : dict
        The problem definition
    X : numpy.array
        A NumPy array containing the model inputs
    Y : numpy.array
        A NumPy array containing the model outputs
    M : int
        The interference parameter, i.e., the number of harmonics to sum in
        the Fourier series decomposition (default 10)
    print_to_console : bool
        Print results directly to console (default False)

    References
    ----------
    .. [1] S. Tarantola, D. Gatelli and T. Mara (2006) "Random Balance Designs
          for the Estimation of First Order Global Sensitivity Indices",
          Reliability Engineering and System Safety, 91:6, 717-727

    .. [2] Elmar Plischke (2010) "An effective algorithm for computing global
          sensitivity indices (EASI) Reliability Engineering & System Safety",
          95:4, 354-360. doi:10.1016/j.ress.2009.11.005

    .. [3] Jean-Yves Tissot, Clémentine Prieur (2012) "Bias correction for the
          estimation of sensitivity indices based on random balance designs.",
          Reliability Engineering and System Safety, Elsevier, 107, 205-213.
          doi:10.1016/j.ress.2012.06.010

    .. [4] Jeanne Goffart, Mickael Rabouille & Nathan Mendes (2015)
          "Uncertainty and sensitivity analysis applied to hygrothermal
          simulation of a brick building in a hot and humid climate",
          Journal of Building Performance Simulation.
          doi:10.1080/19401493.2015.1112430

    Examples
    --------
    >>> X = latin.sample(problem, 1000)
    >>> Y = Ishigami.evaluate(X)
    >>> Si = rbd_fast.analyze(problem, X, Y, print_to_console=False)
    """
    if seed:
        np.random.seed(seed)

    num_vars = problem['num_vars']
    group_names, D = extract_groups(problem)
    N = Y.size

    # Calculate and Output the First Order Value
    Si = ResultDict((k, [None] * num_vars) for k in ['S1'])
    Si['names'] = group_names

    for i in range(num_vars):
        S1 = compute_first_order(permute_outputs(X[:, i], Y), M)
        S1 = unskew_S1(S1, M, N)
        Si['S1'][i] = S1

    if D < num_vars:
        groupings, _ = compute_groups_matrix(problem['groups'])
        Si['S1'] = _group_metric(groupings, Si['S1'])
    
    if print_to_console:
        print(Si.to_df())

    return Si


def permute_outputs(X, Y):
    """
    Permute the output according to one of the inputs as in [_2]

    References
    ----------
    .. [2] Elmar Plischke (2010) "An effective algorithm for computing global
          sensitivity indices (EASI) Reliability Engineering & System Safety",
          95:4, 354-360. doi:10.1016/j.ress.2009.11.005

    """
    permutation_index = np.argsort(X)
    permutation_index = np.concatenate([permutation_index[::2],
                                        permutation_index[1::2][::-1]])
    return Y[permutation_index]


def compute_first_order(permuted_outputs, M):
    _, Pxx = periodogram(permuted_outputs)
    V = np.sum(Pxx[1:])
    D1 = np.sum(Pxx[1: M + 1])
    return D1 / V


def unskew_S1(S1, M, N):
    """
    Unskew the sensivity indice
    (Jean-Yves Tissot, Clémentine Prieur (2012) "Bias correction for the
    estimation of sensitivity indices based on random balance designs.",
    Reliability Engineering and System Safety, Elsevier, 107, 205-213.
    doi:10.1016/j.ress.2012.06.010)
    """
    lamb = (2 * M) / N
    return S1 - lamb / (1 - lamb) * (1 - S1)


def cli_parse(parser):
    parser.add_argument('-X', '--model-input-file',
                        type=str, required=True, help='Model input file')
    return parser


def cli_action(args):
    problem = read_param_file(args.paramfile)
    X = np.loadtxt(args.model_input_file,
                   delimiter=args.delimiter)
    Y = np.loadtxt(args.model_output_file,
                   delimiter=args.delimiter,
                   usecols=(args.column,))
    analyze(problem, X, Y, print_to_console=True, seed=args.seed)


if __name__ == "__main__":
    common_args.run_cli(cli_parse, cli_action)
