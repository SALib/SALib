import math
from sys import exit

import numpy as np
from scipy.stats import norm

from . import common_args
from ..util import read_param_file, ResultDict


def analyze(problem, Y, M=4, num_resamples=100, conf_level=0.95, print_to_console=False, seed=None):
    """Performs the Fourier Amplitude Sensitivity Test (FAST) on model outputs.

    Returns a dictionary with keys 'S1' and 'ST', where each entry is a list of
    size D (the number of parameters) containing the indices in the same order
    as the parameter file.

    Compatible with
    ---------------
    * `fast_sampler`

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

    D = problem['num_vars']

    if Y.size % (D) == 0:
        N = int(Y.size / D)
    else:
        print("""
            Error: Number of samples in model output file must be a multiple of D,
            where D is the number of parameters in your parameter file.
          """)
        exit()

    # Recreate the vector omega used in the sampling
    omega = np.zeros([D])
    omega[0] = math.floor((N - 1) / (2 * M))
    m = math.floor(omega[0] / (2 * M))

    if m >= (D - 1):
        omega[1:] = np.floor(np.linspace(1, m, D - 1))
    else:
        omega[1:] = np.arange(D - 1) % m + 1

    # Calculate and Output the First and Total Order Values
    Si = ResultDict((k, [None] * D) for k in ['S1', 'ST', 'S1_conf', 'ST_conf'])
    Si['names'] = problem['names']

    omega_0 = omega[0]
    for i in range(D):
        l = np.arange(i * N, (i + 1) * N)
        Y_l = Y[l]

        S1, ST = compute_orders(Y_l, N, M, omega_0)
        Si['S1'][i] = S1
        Si['ST'][i] = ST

        S1_d_conf, ST_d_conf = bootstrap(Y_l, N, M, omega_0, num_resamples, conf_level)
        Si['S1_conf'][i] = S1_d_conf
        Si['ST_conf'][i] = ST_d_conf
    
    if print_to_console:
        print(Si.to_df())

    return Si


def compute_orders(outputs, N, M, omega):
    f = np.fft.fft(outputs)
    Sp = np.power(np.absolute(f[np.arange(1, int((N + 1) / 2))]) / N, 2)
    V = 2.0 * np.sum(Sp)

    # Calculate first and total order
    D1 = 2.0 * np.sum(Sp[np.arange(1, M + 1) * int(omega) - 1])
    Dt = 2.0 * np.sum(Sp[np.arange(int(omega / 2.0))])

    return (D1 / V), (1.0 - Dt / V)


def bootstrap(Y, N, M, omega_0, resamples, conf_level):
    # Use half of available data each time
    T_data = Y.shape[0]
    n_size = int(T_data * 0.5)

    res_S1 = np.zeros(resamples)
    res_ST = np.zeros(resamples)
    for i in range(resamples):
        sample_idx = np.random.choice(T_data, replace=True, size=n_size)
        Y_rs = Y[sample_idx]

        S1, ST = compute_orders(Y_rs, N, M, omega_0)
        res_S1[i] = S1
        res_ST[i] = ST

    bnd = norm.ppf(0.5 + conf_level / 2.0)
    S1_conf = bnd * res_S1.std(ddof=1)
    ST_conf = bnd * res_ST.std(ddof=1)
    return S1_conf, ST_conf


# No additional arguments required for FAST
def cli_parse(parser):
    """Add method specific options to CLI parser.

    Parameters
    ----------
    parser : argparse object

    Returns
    ----------
    Updated argparse object
    """
    parser.add_argument('-M', '--M', type=int, required=False,
                        default=4,
                        help='Inference parameter')
    parser.add_argument('-r', '--resamples', type=int, required=False,
                        default=100,
                        help='Number of bootstrap resamples for Sobol '
                        'confidence intervals')

    return parser



def cli_action(args):
    problem = read_param_file(args.paramfile)
    Y = np.loadtxt(args.model_output_file,
                   delimiter=args.delimiter, usecols=(args.column,))

    analyze(problem, Y, M=args.M, num_resamples=args.resamples, 
            print_to_console=True, seed=args.seed)


if __name__ == "__main__":
    common_args.run_cli(cli_parse, cli_action)
