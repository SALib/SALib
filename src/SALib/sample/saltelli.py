from typing import Dict, Optional
import warnings
import numpy as np
import math

from . import common_args
from . import sobol_sequence
from ..util import (scale_samples, read_param_file,
                    compute_groups_matrix, _check_groups)


def sample(problem: Dict, N: int, calc_second_order: bool = True,
           seed: Optional[int] = None, skip_values: Optional[int] = 1024):
    """Generates model inputs using Saltelli's extension of the Sobol' sequence.

    Returns a NumPy matrix containing the model inputs using Saltelli's sampling
    scheme. Saltelli's scheme extends the Sobol' sequence in a way to reduce
    the error rates in the resulting sensitivity index calculations. If
    `calc_second_order` is False, the resulting matrix has ``N * (D + 2)``
    rows, where ``D`` is the number of parameters. If `calc_second_order` is True,
    the resulting matrix has ``N * (2D + 2)`` rows. These model inputs are
    intended to be used with :func:`SALib.analyze.sobol.analyze`.

    Parameters
    ----------
    problem : dict
        The problem definition
    N : int
        The number of samples to generate.
        Must be a power of 2 and <= `skip_values`.
    calc_second_order : bool
        Calculate second-order sensitivities (default True)
    skip_values : int
        Number of points in Sobol' sequence to skip (must be a power of 2).

    References
    ----------
    .. [1] Saltelli, A., 2002.
           Making best use of model evaluations to compute sensitivity indices.
           Computer Physics Communications 145, 280–297.
           https://doi.org/10.1016/S0010-4655(02)00280-1

    .. [2] Sobol', I.M., 2001.
           Global sensitivity indices for nonlinear mathematical models and
           their Monte Carlo estimates.
           Mathematics and Computers in Simulation,
           The Second IMACS Seminar on Monte Carlo Methods 55, 271–280.
           https://doi.org/10.1016/S0378-4754(00)00270-6

    .. [3] Owen, A. B., 2020. 
           On dropping the first Sobol' point. 
           arXiv:2008.08051 [cs, math, stat]. 
           Available at: http://arxiv.org/abs/2008.08051 (Accessed: 20 April 2021).

    .. [4] Discussion: https://github.com/scipy/scipy/pull/10844
    """
    if seed:
        msg = ("The seed value is ignored for the Saltelli sampler\n"
               "as it uses the (deterministic) Sobol' sequence.\n"
               "Different samples can be obtained by setting the\n"
               "`skip_values` parameter (defaults to 1024).")
        warnings.warn(msg)


    # bit-shift test to check if `N` is a power of 2
    if not ((N & (N-1) == 0) and (N != 0 and N-1 != 0)):
        msg = f"""
        Convergence properties of the Sobol' sequence is only valid if
        `N` ({N}) is a power of 2.
        """
        raise ValueError(msg)

    M = skip_values
    if not ((M & (M-1) == 0) and (M != 0 and M-1 != 0)):
        msg = """
        Convergence properties of the Sobol' sequence is only valid if
        `skip_values` ({M}) is a power of 2.
        """
        raise ValueError(msg)

    n_exp = int(math.log(N, 2))
    m_exp = int(math.log(M, 2))
    if n_exp >= m_exp:
        msg = f"Convergence may not be valid as 2^{n_exp} ({N}) is >= 2^{m_exp} ({M})."
        raise ValueError(msg)


    D = problem['num_vars']
    groups = _check_groups(problem)

    if not groups:
        Dg = problem['num_vars']
    else:
        G, group_names = compute_groups_matrix(groups)
        Dg = len(set(group_names))

    # Create base sequence - could be any type of sampling
    base_sequence = sobol_sequence.sample(N + skip_values, 2 * D)

    if calc_second_order:
        saltelli_sequence = np.zeros([(2 * Dg + 2) * N, D])
    else:
        saltelli_sequence = np.zeros([(Dg + 2) * N, D])
    index = 0

    for i in range(skip_values, N + skip_values):

        # Copy matrix "A"
        for j in range(D):
            saltelli_sequence[index, j] = base_sequence[i, j]

        index += 1

        # Cross-sample elements of "B" into "A"
        for k in range(Dg):
            for j in range(D):
                if (not groups and j == k) or (groups and group_names[k] == groups[j]):
                    saltelli_sequence[index, j] = base_sequence[i, j + D]
                else:
                    saltelli_sequence[index, j] = base_sequence[i, j]

            index += 1

        # Cross-sample elements of "A" into "B"
        # Only needed if you're doing second-order indices (true by default)
        if calc_second_order:
            for k in range(Dg):
                for j in range(D):
                    if (not groups and j == k) or (groups and group_names[k] == groups[j]):
                        saltelli_sequence[index, j] = base_sequence[i, j]
                    else:
                        saltelli_sequence[index, j] = base_sequence[i, j + D]

                index += 1

        # Copy matrix "B"
        for j in range(D):
            saltelli_sequence[index, j] = base_sequence[i, j + D]

        index += 1

    saltelli_sequence = scale_samples(saltelli_sequence, problem)
    return saltelli_sequence


def cli_parse(parser):
    """Add method specific options to CLI parser.

    Parameters
    ----------
    parser : argparse object

    Returns
    ----------
    Updated argparse object
    """
    parser.add_argument('--max-order', type=int, required=False, default=2,
                        choices=[1, 2],
                        help='Maximum order of sensitivity indices \
                           to calculate')
    return parser


def cli_action(args):
    """Run sampling method

    Parameters
    ----------
    args : argparse namespace
    """
    problem = read_param_file(args.paramfile)
    param_values = sample(problem, args.samples,
                          calc_second_order=(args.max_order == 2),
                          seed=args.seed)
    np.savetxt(args.output, param_values, delimiter=args.delimiter,
               fmt='%.' + str(args.precision) + 'e')


if __name__ == "__main__":
    common_args.run_cli(cli_parse, cli_action)
