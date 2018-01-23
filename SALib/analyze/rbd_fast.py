#!/usr/bin/env python
# coding=utf8

from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.signal import periodogram

from . import common_args
from ..util import read_param_file


def analyze(problem, Y, X, M=10, print_to_console=False):
    """Performs the Random Balanced Design - Fourier Amplitude Sensitivity Test
    (RBD-FAST) on model outputs.

    Returns a dictionary with keys 'S1', where each entry is a list of
    size D (the number of parameters) containing the indices in the same order
    as the parameter file.

    Parameters
    ----------
    problem : dict
        The problem definition
    Y : numpy.array
        A NumPy array containing the model outputs
    X : numpy.array
        A NumPy array containing the model inputs
    M : int
        The interference parameter, i.e., the number of harmonics to sum in
        the Fourier series decomposition (default 4)
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
    >>> Si = rbd_fast.analyze(problem, Y, X, print_to_console=False)
    """

    D = problem['num_vars']
    N = Y.size

    # Calculate and Output the First Order Value
    if print_to_console:
        print("Parameter First")
    Si = dict((k, [None] * D) for k in ['S1'])
    for i in range(D):
        S1 = compute_first_order(permute_outputs(Y, X[:, i]), M)
        S1 = unskew_S1(S1, M, N)
        Si['S1'][i] = S1
        if print_to_console:
            print("%s %g" %
                  (problem['names'][i].ljust(9), Si['S1'][i]))
    return Si


def permute_outputs(Y, X):
    """
    Permute the output according to one of the inputs
    (Elmar Plischke (2010) "An effective algorithm for computing global
     sensitivity indices (EASI) Reliability Engineering & System Safety",
     95:4, 354-360. doi:10.1016/j.ress.2009.11.005)
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


if __name__ == "__main__":

    parser = common_args.create()
    parser.add_argument('-X', '--model-input-file',
                        type=str, required=True, help='Model input file')
    args = parser.parse_args()
    problem = read_param_file(args.paramfile)
    Y = np.loadtxt(args.model_output_file,
                   delimiter=args.delimiter,
                   usecols=(args.column,))
    X = np.loadtxt(args.model_input_file,
                   delimiter=args.delimiter)

    analyze(problem, Y, X, print_to_console=True)
