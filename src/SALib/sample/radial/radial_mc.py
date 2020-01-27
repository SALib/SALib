
import numpy as np
from .. import sobol_sequence
from SALib.util import scale_samples, read_param_file
from typing import Dict, Optional

from .. import common_args
from .radial_funcs import combine_samples


__all__ = ['sample']


def sample(problem: Dict, N: int,
           seed: Optional[int] = None):
    """Generates `N` monte carlo samples for a Radial OAT approach using a uniform distribution.

    Compatible with:
    * `radial_ee`
    * `sobol_jansen`

    Arguments
    ---------
    problem : dict
        SALib problem specification

    N : int
        The number of sample sets to generate

    seed : int
        Seed value to use for np.random.seed

    Returns
    ---------
    numpy.ndarray : An array of samples


    Usage Example
    -------------
    ```python
    >>> X = sample(problem, N, seed)
    ```
    
    `X` will now hold:
    [
        [x_{1,1}, x_{1,2}, ..., x_{1,p}]
        [b_{1,1}, x_{1,2}, ..., x_{1,p}]
        [x_{1,1}, b_{1,2}, ..., x_{1,p}]
        [x_{1,1}, x_{1,2}, ..., b_{1,p}]
        ...
        [x_{N,1}, x_{N,2}, ..., x_{N,p}]
        [b_{N,1}, x_{N,2}, ..., x_{N,p}]
        [x_{N,1}, b_{N,2}, ..., x_{N,p}]
        [x_{N,1}, x_{N,2}, ..., b_{N,p}]
    ]

    where `p` denotes the number of parameters as specified in `problem` and
    `b` represents perturbed values.

    The first parameter set in each sample set acts as the baseline.

    We can now run the model using the values in `X`. 
    The total number of model evaluations will be `N(p+1)`.


    References
    ----------
    .. [1] Campolongo, F., Saltelli, A., Cariboni, J., 2011. 
           From screening to quantitative sensitivity analysis: A unified approach. 
           Computer Physics Communications 182, 978–988.
           https://www.sciencedirect.com/science/article/pii/S0010465510005321
           DOI: 10.1016/j.cpc.2010.12.039
    """
    if seed:
        np.random.seed(seed)

    num_vars = problem['num_vars']
    bounds = problem['bounds']

    # Generate the 'nominal' parameter positions
    # Total number of parameter sets = N*(p+1)
    min_bnds = [lb[0] for lb in bounds]
    max_bnds = [lb[1] for lb in bounds]
    sequence = np.random.uniform(min_bnds, max_bnds, size=(N, num_vars*2))

    # Use first N cols as baseline points
    baseline = sequence[:, :num_vars]
    scale_samples(baseline, bounds)

    # Use next N cols as perturbation points
    perturb = sequence[:, num_vars:]
    scale_samples(perturb, bounds)

    sample_set = combine_samples(baseline, perturb)

    return sample_set


# No additional CLI options
cli_parse = None


def cli_action(args):
    """Run sampling method

    Parameters
    ----------
    args : argparse namespace
    """
    problem = read_param_file(args.paramfile)

    param_values = sample(problem, args.samples, seed=args.seed)
    np.savetxt(args.output, param_values, delimiter=args.delimiter,
               fmt='%.' + str(args.precision) + 'e')

if __name__ == '__main__':
    common_args.run_cli(cli_parse, cli_action)