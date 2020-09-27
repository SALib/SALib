from typing import Dict

import numpy as np

from . import common_args
from . import sobol_sequence
from ..util import scale_samples, read_param_file


def sample(problem: Dict, N: int, delta: float = 0.01, 
           seed: int = None, skip_values: int = 1000) -> np.ndarray:
    """Generate matrix of samples for derivative-based global sensitivity measure (dgsm).
    Start from a QMC (sobol) sequence and finite difference with delta % steps

    Parameters
    ----------
    problem : dict
        SALib problem specification

    N : int
        Number of samples

    delta : float
        Finite difference step size (percent)
    
    seed : int or None
        Random seed value

    skip_values : int
        How many values of the Sobol sequence to skip

    Returns
    ----------
    np.array : DGSM sequence

    References
    ----------
    .. [1] Sobol', I.M., Kucherenko, S., 2009. 
           Derivative based global sensitivity measures and their link with 
           global sensitivity indices. 
           Mathematics and Computers in Simulation 79, 3009–3017. 
           https://doi.org/10.1016/j.matcom.2009.01.023

    .. [2] Sobol', I.M., Kucherenko, S., 2010. 
           Derivative based global sensitivity measures. 
           Procedia - Social and Behavioral Sciences 2, 7745–7746. 
           https://doi.org/10.1016/j.sbspro.2010.05.208

    """
    if seed:
        np.random.seed(seed)

    D = problem['num_vars']
    bounds = problem['bounds']

    # Create base sequence - could be any type of sampling
    base_sequence = sobol_sequence.sample(N + skip_values, D)

    # scale before finite differencing
    base_sequence = scale_samples(base_sequence, problem)

    dgsm_sequence = np.empty([N * (D + 1), D])

    index = 0
    for i in range(skip_values, N + skip_values):
        # Copy the initial point
        dgsm_sequence[index, :] = base_sequence[i, :]

        index += 1
        for j in range(D):
            temp = np.zeros(D)
            bnd_j = bounds[j]
            temp[j] = base_sequence[i, j] * delta
            dgsm_sequence[index, :] = base_sequence[i, :] + temp
            dgsm_sequence[index, j] = min(
                dgsm_sequence[index, j], bnd_j[1])
            dgsm_sequence[index, j] = max(
                dgsm_sequence[index, j], bnd_j[0])
            index += 1

    return dgsm_sequence


def cli_parse(parser):
    """Add method specific options to CLI parser.

    Parameters
    ----------
    parser : argparse object

    Returns
    ----------
    Updated argparse object
    """
    parser.add_argument('-d', '--delta', type=float,
                        required=False, default=0.01,
                        help='Finite difference step size (percent)')
    return parser


def cli_action(args):
    """Run sampling method

    Parameters
    ----------
    args : argparse namespace
    """
    problem = read_param_file(args.paramfile)
    param_values = sample(problem, args.samples, args.delta, seed=args.seed)
    np.savetxt(args.output, param_values, delimiter=args.delimiter,
               fmt='%.' + str(args.precision) + 'e')


if __name__ == "__main__":
    common_args.run_cli(cli_parse, cli_action)
