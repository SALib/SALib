
from typing import Dict, Optional
import numpy as np

from SALib.util import scale_samples, read_param_file
from .. import sobol_sequence
from SALib.sample import common_args
from .radial_funcs import combine_samples



__all__ = ['sample']


def sample(problem: Dict, N: int, 
           skip_num=0,
           seed: Optional[int] = None):
    """Generates `N` sobol samples for a Radial OAT approach based on sobol sequences.

    Results can be analyzed using 
    * `sobol_jansen`
    * `radial_ee`

    References
    ----------
    .. [1] Campolongo, F., Saltelli, A., Cariboni, J., 2011. 
           From screening to quantitative sensitivity analysis: A unified approach. 
           Computer Physics Communications 182, 978–988.
           https://www.sciencedirect.com/science/article/pii/S0010465510005321
           DOI: 10.1016/j.cpc.2010.12.039

    Arguments
    ---------
    problem : dict
        SALib problem specification

    N : int
        The number of sample sets to generate

    skip_num : int
        Number of sobol sequence values to skip
        When conducting a sequential sensitivity analysis, this is the 
        previous number of samples used

    seed : int
        Seed value to use for np.random.seed

    Example
    -------
    >>> X = sample(problem, N, seed)
    
    `X` will now hold:
    [[x_{1,1}, x_{1,2}, ..., x_{1,p}]
    [b_{1,1}, x_{1,2}, ..., x_{1,p}]
    [x_{1,1}, b_{1,2}, ..., x_{1,p}]
    [x_{1,1}, x_{1,2}, ..., b_{1,p}]
    ...
    [x_{N,1}, x_{N,2}, ..., x_{N,p}]
    [b_{N,1}, x_{N,2}, ..., x_{N,p}]
    [x_{N,1}, b_{N,2}, ..., x_{N,p}]
    [x_{N,1}, x_{N,2}, ..., b_{N,p}]]

    where `p` denotes the number of parameters as
    specified in `problem`

    We can now run the model using the values in `X`. 
    The total number of model evaluations will be `N(p+1)`.

    Returns
    ---------
    numpy.ndarray : An array of samples
    """
    if seed:
        np.random.seed(seed)

    num_vars = problem['num_vars']
    bounds = problem['bounds']

    # "we need to generate a quasi-random matrix of Sobol’ numbers of 
    # size (R,2k), with R > r"
    # "We obtain good results by systematically discarding 
    # four points (R = r + 4)" (Campolongo et al. 2011, p 981)
    # 
    # In this context, `N := r`
    # Given that R > N, we create an array of (N*2, `num_vars`*2)
    # and use the second half of the array (N:, num_vars:)
    # as the perturbation set.

    skip_N = skip_num * 2
    sequence = sobol_sequence.sample(skip_N+(N*2), (num_vars*2))
    sequence = sequence[skip_N:, :]

    # Use first N rows and `num_vars` cols as baseline points
    # and next `num_vars` cols and N rows as perturbation points
    baseline = sequence[:N, :num_vars]
    scale_samples(baseline, bounds)
    
    perturb = sequence[N:, num_vars:]
    scale_samples(perturb, bounds)

    # Total number of parameter sets = `N*(num_vars+1)`
    sample_set = combine_samples(baseline, perturb)

    return sample_set


def cli_parse(parser):
    parser.add_argument('-k', '--skip_num', type=int, required=False,
                        default=0,
                        help='Number of Sobol values to skip (default 0)')
    return parser


def cli_action(args):
    """Run sampling method

    Parameters
    ----------
    args : argparse namespace
    """
    problem = read_param_file(args.paramfile)

    param_values = sample(problem, args.samples, skip_num=args.skip_num, 
                          seed=args.seed)
    np.savetxt(args.output, param_values, delimiter=args.delimiter,
               fmt='%.' + str(args.precision) + 'e')

if __name__ == '__main__':
    common_args.run_cli(cli_parse, cli_action)