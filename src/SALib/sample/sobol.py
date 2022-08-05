from typing import Dict

import numpy as np
from scipy.stats import qmc

from . import common_args
from ..util import (scale_samples, read_param_file,
                    compute_groups_matrix, _check_groups)


def sample(
    problem: Dict,
    N: int,
    *,
    calc_second_order: bool = True,
    scramble: bool = True
):
    """Generates model inputs using Saltelli's extension of the Sobol' sequence

    The Sobol' sequence is a popular quasi-random low-discrepancy sequence used
    to generate uniform samples of parameter space.

    Returns a NumPy matrix containing the model inputs using Saltelli's
    sampling scheme.
    Saltelli's scheme extends the Sobol' sequence in a way to reduce
    the error rates in the resulting sensitivity index calculations. If
    `calc_second_order` is False, the resulting matrix has ``N * (D + 2)``
    rows, where ``D`` is the number of parameters.
    If `calc_second_order` is `True`, the resulting matrix has ``N * (2D + 2)``
    rows.
    These model inputs are intended to be used with
    :func:`SALib.analyze.sobol.analyze`.

    Notes
    -----
    The initial points of the Sobol' sequence has some repetition (see Table 2
    in Campolongo [1]_), which can be avoided by setting the `skip_values`
    parameter. Skipping values reportedly improves the uniformity of samples.
    It has been shown that naively skipping values may reduce accuracy,
    increasing the number of samples needed to achieve convergence
    (see Owen [2]_).

    Parameters
    ----------
    problem : dict
        The problem definition
    N : int
        The number of samples to generate.
        Ideally a power of 2 and <= `skip_values`.
    calc_second_order : bool
        Calculate second-order sensitivities (default True)
    scramble : bool
        Default is True.

    References
    ----------
    .. [1] Campolongo, F., Saltelli, A., Cariboni, J., 2011.
           From screening to quantitative sensitivity analysis.
           A unified approach.
           Computer Physics Communications 182, 978–988.
           https://doi.org/10.1016/j.cpc.2010.12.039

    .. [2] Owen, A. B., 2020.
           On dropping the first Sobol' point.
           arXiv:2008.08051 [cs, math, stat].
           Available at: http://arxiv.org/abs/2008.08051
           (Accessed: 20 April 2021).

    .. [3] Saltelli, A., 2002.
           Making best use of model evaluations to compute sensitivity indices.
           Computer Physics Communications 145, 280–297.
           https://doi.org/10.1016/S0010-4655(02)00280-1

    .. [4] Sobol', I.M., 2001.
           Global sensitivity indices for nonlinear mathematical models and
           their Monte Carlo estimates.
           Mathematics and Computers in Simulation,
           The Second IMACS Seminar on Monte Carlo Methods 55, 271–280.
           https://doi.org/10.1016/S0378-4754(00)00270-6
    """
    D = problem['num_vars']
    groups = _check_groups(problem)

    # Create base sequence - could be any type of sampling
    qrng = qmc.Sobol(d=2*D, scramble=scramble)
    base_sequence = qrng.random(N)

    if not groups:
        Dg = problem['num_vars']
    else:
        G, group_names = compute_groups_matrix(groups)
        Dg = len(set(group_names))

    if calc_second_order:
        saltelli_sequence = np.zeros([(2 * Dg + 2) * N, D])
    else:
        saltelli_sequence = np.zeros([(Dg + 2) * N, D])
    index = 0

    for i in range(N):
        # Copy matrix "A"
        for j in range(D):
            saltelli_sequence[index, j] = base_sequence[i, j]

        index += 1

        # Cross-sample elements of "B" into "A"
        for k in range(Dg):
            for j in range(D):
                if (not groups and j == k) or \
                   (groups and group_names[k] == groups[j]):
                    saltelli_sequence[index, j] = base_sequence[i, j + D]
                else:
                    saltelli_sequence[index, j] = base_sequence[i, j]

            index += 1

        # Cross-sample elements of "A" into "B"
        # Only needed if you're doing second-order indices (true by default)
        if calc_second_order:
            for k in range(Dg):
                for j in range(D):
                    if (not groups and j == k) or \
                       (groups and group_names[k] == groups[j]):
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
    parser.add_argument(
        '--max-order', type=int, required=False, default=2, choices=[1, 2],
        help='Maximum order of sensitivity indices to calculate'
    )
    parser.add_argument(
        '--scramble', type=int, required=False, default=True,
        help='Use scrambled sequence'
    )

    # hacky way to remove an argument (seed option not relevant for Saltelli)
    remove_opts = [x for x in parser._actions if x.dest == 'seed']
    [parser._handle_conflict_resolve(None, [('--seed', x), ('-s', x)])
     for x in remove_opts]

    return parser


def cli_action(args):
    """Run sampling method

    Parameters
    ----------
    args : argparse namespace
    """
    problem = read_param_file(args.paramfile)
    param_values = sample(
        problem, args.samples,
        calc_second_order=(args.max_order == 2),
        scramble=args.scramble
    )
    np.savetxt(args.output, param_values, delimiter=args.delimiter,
               fmt='%.' + str(args.precision) + 'e')


if __name__ == "__main__":
    common_args.run_cli(cli_parse, cli_action)
