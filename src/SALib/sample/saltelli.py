import warnings
import numpy as np

from . import common_args
from . import sobol_sequence
from ..util import (scale_samples, read_param_file, 
                    extract_groups)


def sample(problem, N, calc_second_order=True, seed=None, skip_values=1000):
    """Generates model inputs using Saltelli's extension of the Sobol sequence.

    Returns a NumPy matrix containing the model inputs using Saltelli's sampling
    scheme.  Saltelli's scheme extends the Sobol sequence in a way to reduce
    the error rates in the resulting sensitivity index calculations.  If
    calc_second_order is False, the resulting matrix has N * (D + 2)
    rows, where D is the number of parameters.  If calc_second_order is True,
    the resulting matrix has N * (2D + 2) rows.  These model inputs are
    intended to be used with :func:`SALib.analyze.sobol.analyze`.

    Parameters
    ----------
    problem : dict
        The problem definition
    N : int
        The number of samples to generate
    calc_second_order : bool
        Calculate second-order sensitivities (default True)
    """
    if seed:
        msg = "The seed value is ignored for the Saltelli sampler\n"
        msg += "as it uses the (deterministic) Sobol sequence.\n"
        msg += "Different samples can be obtained by setting the\n"
        msg += "`skip_values` parameter (defaults to 1000)."
        warnings.warn(msg)

    D = problem['num_vars']
    groups = problem.get('groups')
    group_names, Dg = extract_groups(problem)

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
