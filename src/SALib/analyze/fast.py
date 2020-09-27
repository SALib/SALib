import math
from sys import exit

import numpy as np

from . import common_args
from ..util import (read_param_file, ResultDict, extract_groups, 
                    compute_groups_matrix, _group_metric)


def analyze(problem, Y, M=4, print_to_console=False, seed=None):
    """Performs the Fourier Amplitude Sensitivity Test (FAST) on model outputs.

    Returns a dictionary with keys 'S1' and 'ST', where each entry is a list of
    size D (the number of parameters) containing the indices in the same order
    as the parameter file.

    Parameters
    ----------
    problem : dict
        The problem definition
    Y : numpy.array
        A NumPy array containing the model outputs
    M : int
        The interference parameter, i.e., the number of harmonics to sum in
        the Fourier series decomposition (default 4)
    print_to_console : bool
        Print results directly to console (default False)

    References
    ----------
    .. [1] Cukier, R. I., C. M. Fortuin, K. E. Shuler, A. G. Petschek, and J. H.
           Schaibly (1973).  "Study of the sensitivity of coupled reaction
           systems to uncertainties in rate coefficients."  J. Chem. Phys.,
           59(8):3873-3878, doi:10.1063/1.1680571.

    .. [2] Saltelli, A., S. Tarantola, and K. P.-S. Chan (1999).  "A
          Quantitative Model-Independent Method for Global Sensitivity
          Analysis of Model Output."  Technometrics, 41(1):39-56,
          doi:10.1080/00401706.1999.10485594.

    Examples
    --------
    >>> X = fast_sampler.sample(problem, 1000)
    >>> Y = Ishigami.evaluate(X)
    >>> Si = fast.analyze(problem, Y, print_to_console=False)
    """
    if seed:
        np.random.seed(seed)

    num_vars = problem['num_vars']
    group_names, D = extract_groups(problem)

    if Y.size % (num_vars) == 0:
        N = int(Y.size / num_vars)
    else:
        print("""
            Error: Number of samples in model output file must be a multiple of D,
            where D is the number of parameters in your parameter file.
          """)
        exit()

    # Recreate the vector omega used in the sampling
    omega = np.zeros([num_vars])
    omega[0] = math.floor((N - 1) / (2 * M))
    m = math.floor(omega[0] / (2 * M))

    if m >= (num_vars - 1):
        omega[1:] = np.floor(np.linspace(1, m, num_vars - 1))
    else:
        omega[1:] = np.arange(num_vars - 1) % m + 1

    # Calculate and Output the First and Total Order Values
    keys = ('S1', 'ST')
    Si = ResultDict((k, np.zeros(num_vars)) for k in keys)
    Si['names'] = group_names

    for i in range(num_vars):
        l = np.arange(i * N, (i + 1) * N)
        Si['S1'][i] = compute_first_order(Y[l], N, M, omega[0])
        Si['ST'][i] = compute_total_order(Y[l], N, omega[0])

    if D < num_vars:
        groupings, _ = compute_groups_matrix(problem['groups'])
        for key in keys:
            Si[key] = _group_metric(groupings, Si[key])
    
    if print_to_console:
        print(Si.to_df())

    return Si


def compute_first_order(outputs, N, M, omega):
    f = np.fft.fft(outputs)
    Sp = np.power(np.absolute(f[np.arange(1, int((N + 1) / 2))]) / N, 2)
    V = 2 * np.sum(Sp)
    D1 = 2 * np.sum(Sp[np.arange(1, M + 1) * int(omega) - 1])
    return D1 / V


def compute_total_order(outputs, N, omega):
    f = np.fft.fft(outputs)
    Sp = np.power(np.absolute(f[np.arange(1, int((N + 1) / 2))]) / N, 2)
    V = 2 * np.sum(Sp)
    Dt = 2 * sum(Sp[np.arange(int(omega / 2))])
    return (1 - Dt / V)


# No additional arguments required for FAST
cli_parse = None


def cli_action(args):
    problem = read_param_file(args.paramfile)
    Y = np.loadtxt(args.model_output_file,
                   delimiter=args.delimiter, usecols=(args.column,))

    analyze(problem, Y, print_to_console=True, seed=args.seed)


if __name__ == "__main__":
    common_args.run_cli(cli_parse, cli_action)
