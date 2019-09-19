
import numpy as np
from . import sobol_sequence
from SALib.util import scale_samples
from typing import Dict, Optional

__all__ = ['sample']


def sample(problem: Dict, N: int, 
           seed: Optional[int] = None):
    """Generates `N` samples for a radial OAT approach.

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
        the number of sample sets to generate

    seed : int
        the number of parameters

    Returns
    ---------
    numpy.ndarray : An array of samples
    """
    if seed:
        np.random.seed(seed)

    p = problem['num_vars']
    bounds = problem['bounds']

    # Generate the "nominal" values and their perturbations.
    # "We obtain good results by systematically discarding four points (R = r + 4)" (p 5)
    # in this context, `r := N`
    discard = 4
    R = N + discard

    # Generate the 'nominal' parameter positions
    # Total number of parameter sets = N*(p+1)
    base_sequence = sobol_sequence.sample(R, p)
    subsetted_base = base_sequence[discard:]
    scale_samples(subsetted_base, bounds)

    group = p+1
    sample_set = np.repeat(subsetted_base, repeats=group, axis=0)

    perturbations = sobol_sequence.sample(R+N, p)
    perturbations = perturbations[R:]
    scale_samples(perturbations, bounds)

    grp_start = 0
    for i in range(perturbations.shape[0]):
        mod = np.diag(perturbations[i])

        np.copyto(sample_set[grp_start+1:grp_start+p+1], mod, where=mod != 0.0)
        grp_start += p+1
    # End for

    # The above will look like:
    # [x_{1,1}, x_{1,2}, ..., x_{1,p}]
    # [b_{1,1}, x_{1,2}, ..., x_{1,p}]
    # [x_{1,1}, b_{1,2}, ..., x_{1,p}]
    # [x_{1,1}, x_{1,2}, ..., b_{1,p}]
    # ...
    # [x_{N,1}, x_{N,2}, ..., x_{N,p}]
    # [b_{N,1}, x_{N,2}, ..., x_{N,p}]
    # [x_{N,1}, b_{N,2}, ..., x_{N,p}]
    # [x_{N,1}, x_{N,2}, ..., b_{N,p}]

    # We can now run the model using the values from the
    # sample set. The total number of model evaluations
    # will be `N(p+1)`.

    return sample_set
