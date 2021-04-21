from typing import Dict, Optional
import warnings
import numpy as np
import math

from . import common_args
from . import sobol_sequence
from ..util import (scale_samples, read_param_file,
                    compute_groups_matrix, _check_groups)


def convergence_issue_msg(msg, check_conv):
    if check_conv:
        raise ValueError(msg)
    else:
        warnings.warn(msg)


def sample(problem: Dict, N: int, calc_second_order: bool = True,
           skip_values: int = 1024, check_conv: bool = True):
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
        Must be an exponent of 2 and < `skip_values`.
    calc_second_order : bool
        Calculate second-order sensitivities (default True)
    skip_values : int
        Number of points in Sobol' sequence to skip (must be an exponent of 2).
    check_conv : bool (default: True)
        Check that `N` and `skip_values` meet sample size requirements.
        Convergence properties of the Sobol' sequence is only valid when
        ``N < skip_values`` and that both are exponents of 2 (e.g., `N = 2^n`).
        If True, raises error if this condition is not met.

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
    # bit-shift test to check if `N` == 2**n
    if not ((N & (N-1) == 0) and (N != 0 and N-1 != 0)):
        msg = f"""
        Convergence properties of the Sobol' sequence is only valid if
        `N` ({N}) is equal to `2^n`.
        """
        convergence_issue_msg(msg, check_conv)

    M = skip_values
    if not ((M & (M-1) == 0) and (M != 0 and M-1 != 0)):
        msg = f"""
        Convergence properties of the Sobol' sequence is only valid if
        `skip_values` ({M}) is equal to `2^m`.
        """
        convergence_issue_msg(msg, check_conv)

    if check_conv:
        n_exp = int(math.log(N, 2))
        m_exp = int(math.log(M, 2))
        if n_exp >= m_exp:
            msg = f"Convergence may not be valid as 2^{n_exp} ({N}) is >= 2^{m_exp} ({M})."
            convergence_issue_msg(msg, check_conv)

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
    parser.add_argument('--skip-values', type=int, required=False, default=1024,
                        help='Number of sample points to skip (default: 1024)')
    parser.add_argument('--check-conv', type=bool, required=False, default=True,
                        help='Check samples meet convergence property requirements (default: True)')

    # hacky way to remove an argument (seed option is not relevant for Saltelli)
    remove_opts = [x for x in parser._actions if x.dest == 'seed']
    [parser._handle_conflict_resolve(None, [('--seed', x), ('-s', x)]) for x in remove_opts]

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
                          skip_values=args.skip_values,
                          check_conv=args.check_conv)
    np.savetxt(args.output, param_values, delimiter=args.delimiter,
               fmt='%.' + str(args.precision) + 'e')


if __name__ == "__main__":
    common_args.run_cli(cli_parse, cli_action)
