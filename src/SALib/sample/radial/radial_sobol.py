
import numpy as np
from .. import sobol_sequence
from SALib.util import scale_samples
from typing import Dict, Optional

from .radial_funcs import combine_samples

__all__ = ['sample']


def sample(problem: Dict, N: int, 
           seed: Optional[int] = None):
    """Generates `N` sobol samples for a radial OAT approach.

    Campolongo, F., Saltelli, A., Cariboni, J., 2011. 
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

    sequence = sobol_sequence.sample(N*2, (num_vars*2))

    # Use first N rows and `num_vars` cols as baseline points
    # and next `num_vars` cols and N rows as perturbation points
    baseline = sequence[:N, :num_vars]
    scale_samples(baseline, bounds)
    
    perturb = sequence[N:, num_vars:]
    scale_samples(perturb, bounds)

    # Total number of parameter sets = `N*(num_vars+1)`
    sample_set = combine_samples(baseline, perturb)

    return sample_set