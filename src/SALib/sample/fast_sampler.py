import math

import numpy as np

from . import common_args
from .. util import scale_samples, read_param_file


def sample(problem, N, M=4, seed=None):
    """Generate model inputs for extended Fourier Amplitude Sensitivity Test

    Returns a NumPy matrix containing the model inputs required by the extended
    Fourier Amplitude sensitivity test.  The resulting matrix contains N * D
    rows and D columns, where D is the number of parameters.
    The samples generated are intended to be used by
    :func:`SALib.analyze.fast.analyze`.

    Parameters
    ----------
    problem : dict
        The problem definition
    N : int
        The number of samples to generate
    M : int
        The interference parameter, i.e., the number of harmonics to sum in the
        Fourier series decomposition (default 4)
    seed : int
        Seed to generate a random number

    References
    ----------
    .. [1] Cukier, R.I., Fortuin, C.M., Shuler, K.E., Petschek, A.G.,
           Schaibly, J.H., 1973.
           Study of the sensitivity of coupled reaction systems to
           uncertainties in rate coefficients. I theory.
           Journal of Chemical Physics 59, 3873–3878.
           https://doi.org/10.1063/1.1680571

    .. [2] Saltelli, A., S. Tarantola, and K. P.-S. Chan (1999).  "A
           Quantitative Model-Independent Method for Global Sensitivity
           Analysis of Model Output."  Technometrics, 41(1):39-56,
           doi:10.1080/00401706.1999.10485594.
    """
    if seed:
        np.random.seed(seed)

    if N <= 4 * M**2:
        raise ValueError("""
        Sample size N > 4M^2 is required. M=4 by default.""")

    D = problem['num_vars']

    omega = np.zeros([D])
    omega[0] = math.floor((N - 1) / (2 * M))
    m = math.floor(omega[0] / (2 * M))

    if m >= (D - 1):
        omega[1:] = np.floor(np.linspace(1, m, D - 1))
    else:
        omega[1:] = np.arange(D - 1) % m + 1

    # Discretization of the frequency space, s
    s = (2 * math.pi / N) * np.arange(N)

    # Transformation to get points in the X space
    X = np.zeros([N * D, D])
    omega2 = np.zeros([D])

    for i in range(D):
        omega2[i] = omega[0]
        idx = list(range(i)) + list(range(i + 1, D))
        omega2[idx] = omega[1:]
        z = range(i * N, (i + 1) * N)

        # random phase shift on [0, 2pi) following Saltelli et al.
        # Technometrics 1999
        phi = 2 * math.pi * np.random.rand()

        for j in range(D):
            g = 0.5 + (1 / math.pi) * np.arcsin(np.sin(omega2[j] * s + phi))
            X[z, j] = g

    X = scale_samples(X, problem)

    return X


def cli_parse(parser):
    """Add method specific options to CLI parser.

    Parameters
    ----------
    parser : argparse object

    Returns
    ----------
    Updated argparse object
    """
    parser.add_argument('-M', '--m-coef', type=int, required=False, default=4,
                        help='M coefficient, default 4', dest='M')

    return parser


def cli_action(args):
    """Run sampling method

    Parameters
    ----------
    args : argparse namespace
    """
    problem = read_param_file(args.paramfile)
    param_values = sample(problem, N=args.samples, M=args.M, seed=args.seed)
    np.savetxt(args.output, param_values, delimiter=args.delimiter,
               fmt='%.' + str(args.precision) + 'e')


if __name__ == "__main__":
    common_args.run_cli(cli_parse, cli_action)
