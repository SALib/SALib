from typing import Dict

import numpy as np
from scipy.stats import qmc

from SALib.analyze import common_args
from SALib.util import read_param_file, ResultDict


def analyze(
    problem: Dict,
    X: np.ndarray,
    Y: np.ndarray,
    method: str = "WD",
    print_to_console: bool = False,
    seed: int = None,
):
    """Discrepancy indices.

    Parameters
    ----------
    problem : dict
        The problem definition
    X : numpy.ndarray
        A NumPy array containing the model inputs
    Y : numpy.ndarray
        A NumPy array containing the model outputs
    method : str
        Type of discrepancy, can be 'CD', 'WD', 'MD' or 'L2-star'.
        Refer to `scipy.stats.qmc.discrepancy` for more details. Default is WD.
    print_to_console : bool
        Print results directly to console (default False)
    seed : int
        Seed value to ensure deterministic results
        Unused, but defined to maintain compatibility.

    Notes
    -----
    Compatible with:
        all samplers

    References
    ----------
    1. A. Puy, P.T. Roy and A. Saltelli. 2023. Discrepancy measures for
    sensitivity analysis. https://arxiv.org/abs/2206.13470

    2. A. Saltelli, M. Ratto, T. Andres, F. Campolongo, J. Cariboni, D. Gatelli,
    M. Saisana, and S. Tarantola. 2008.
    Global Sensitivity Analysis: The Primer.
    Wiley, West Sussex, U.K.
    https://dx.doi.org/10.1002/9780470725184
    Accessible at:
    http://www.andreasaltelli.eu/file/repository/Primer_Corrected_2022.pdf

    Examples
    --------

        >>> import numpy as np
        >>> from SALib.sample import latin
        >>> from SALib.analyze import discrepancy
        >>> from SALib.test_functions import Ishigami

        >>> problem = {
        ...   'num_vars': 3,
        ...   'names': ['x1', 'x2', 'x3'],
        ...   'bounds': [[-np.pi, np.pi]]*3
        ... }
        >>> X = latin.sample(problem, 1000)
        >>> Y = Ishigami.evaluate(X)
        >>> Si = discrepancy.analyze(problem, X, Y, print_to_console=True)

    """
    D = problem["num_vars"]
    Y = Y.reshape(-1, 1)

    bounds = np.asarray(problem["bounds"]).T
    X = qmc.scale(sample=X, l_bounds=bounds[0], u_bounds=bounds[1], reverse=True)

    Y = qmc.scale(sample=Y, l_bounds=np.min(Y), u_bounds=np.max(Y), reverse=True)

    s_discrepancy = [
        qmc.discrepancy(np.concatenate([X[:, i, None], Y], axis=1), method=method)
        for i in range(D)
    ]

    s_discrepancy = s_discrepancy / np.sum(s_discrepancy)

    keys = ("s_discrepancy",)
    Si = ResultDict((k, np.zeros(D)) for k in keys)
    Si["names"] = problem["names"]

    Si["s_discrepancy"] = s_discrepancy

    if print_to_console:
        print(Si.to_df())

    return Si


def cli_parse(parser):
    parser.add_argument(
        "-X", "--model-input-file", type=str, required=True, help="Model input file"
    )

    return parser


def cli_action(args):
    problem = read_param_file(args.paramfile)
    X = np.loadtxt(args.model_input_file, delimiter=args.delimiter)
    Y = np.loadtxt(
        args.model_output_file, delimiter=args.delimiter, usecols=(args.column,)
    )
    analyze(
        problem,
        X,
        Y,
        print_to_console=True,
        seed=args.seed,
    )


if __name__ == "__main__":
    common_args.run_cli(cli_parse, cli_action)
