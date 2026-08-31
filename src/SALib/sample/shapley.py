from numbers import Integral
from typing import Optional, Union

import numpy as np

from . import common_args
from ..util import handle_seed, read_param_file, scale_samples


def sample(
    problem: dict,
    N: int,
    seed: Optional[Union[int, np.random.Generator]] = None,
) -> np.ndarray:
    """Generate inputs for Goda's Shapley effects estimator.

    The method draws two independent input vectors for each Monte Carlo
    replication.  A random permutation then defines a trajectory that replaces
    one coordinate at a time from the first vector with the corresponding
    coordinate from the second.  The resulting matrix contains ``N * (D + 1)``
    rows and ``D`` columns.

    Results from these model inputs are intended to be used with
    :func:`SALib.analyze.shapley.analyze`.

    Notes
    -----
    Goda's estimator assumes mutually independent input variables. SALib
    problem definitions specify marginal distributions, so this sampler draws
    every input independently. Grouped parameters are not supported.

    Parameters
    ----------
    problem : dict
        The problem definition.
    N : int
        Number of independent Monte Carlo trajectories. The model is evaluated
        ``N * (D + 1)`` times.
    seed : {None, int, `numpy.random.Generator`}, optional
        Seed or random generator used to draw the sample and permutations.

    Returns
    -------
    numpy.ndarray
        Model inputs arranged as ``N`` consecutive trajectories.

    References
    ----------
    1. Goda, T. (2021). A simple algorithm for global sensitivity analysis
       with Shapley effects. Reliability Engineering & System Safety, 213,
       107702. https://doi.org/10.1016/j.ress.2021.107702
    """
    if isinstance(N, bool) or not isinstance(N, Integral) or N <= 0:
        raise ValueError("N must be a positive integer.")

    _validate_ungrouped(problem)

    num_vars = problem["num_vars"]
    if num_vars <= 0:
        raise ValueError("The problem must contain at least one variable.")

    rng = handle_seed(seed)
    independent_pairs = scale_samples(rng.random((2 * N, num_vars)), problem)
    first, second = np.split(independent_pairs, 2)

    samples = np.empty((N * (num_vars + 1), num_vars), dtype=float)
    for trajectory, permutation in enumerate(
        (rng.permutation(num_vars) for _ in range(N))
    ):
        start = trajectory * (num_vars + 1)
        samples[start] = first[trajectory]

        current = first[trajectory].copy()
        for step, factor in enumerate(permutation, start=1):
            current[factor] = second[trajectory, factor]
            samples[start + step] = current

    return samples


def _validate_ungrouped(problem: dict) -> None:
    groups = problem.get("groups")
    if groups and list(groups) != list(problem.get("names", [])):
        raise ValueError("Goda's Shapley estimator does not support groups.")


# No additional CLI options
cli_parse = None


def cli_action(args):
    """Generate a Shapley trajectory design from command-line arguments."""
    problem = read_param_file(args.paramfile)
    param_values = sample(problem, args.samples, seed=args.seed)
    np.savetxt(
        args.output,
        param_values,
        delimiter=args.delimiter,
        fmt="%." + str(args.precision) + "e",
    )


if __name__ == "__main__":
    common_args.run_cli(cli_parse, cli_action)
