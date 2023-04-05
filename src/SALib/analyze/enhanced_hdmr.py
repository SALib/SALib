import math
import numpy as np
from typing import Dict
from itertools import combinations as comb
from collections import defaultdict, namedtuple





def analyze(
    problem: Dict,
    X: np.ndarray,
    Y: np.ndarray,
    max_order: int = 2,
    poly_order: int = 2,
    bootstrap: int = 20,
    subset: int = None,
    alpha: float = 0.95,
    extended_base: bool = True,
    print_to_console: bool = False,
    seed: int = None
) -> Dict:
    
    # Random Seed
    if seed:
        np.random.seed(seed)

    # Make sure Y, output array, is a matrix
    Y = Y.reshape(-1, 1)
    # Check arguments
    _check_args(X, Y, max_order, poly_order, bootstrap, subset, alpha)
    # Instantiate Core Parameters
    hdmr = _core_params(*X.shape, poly_order, max_order, bootstrap, subset, extended_base)


def _check_args(X, Y, max_order, poly_order, bootstrap, subset, alpha):
    """Perform checks to ensure all parameters are within usable/expected ranges."""
    # Get dimensionality of numpy arrays
    N, d = X.shape
    y_row = Y.shape[0]  

    # Now check input-output mismatch
    if d == 1:
        raise RuntimeError(
            "Matrix X contains only a single column: No point to do"
            " sensitivity analysis when d = 1."
        )
    
    if N < 300:
        raise RuntimeError(
            f"Number of samples in the input matrix X, {N}, is insufficient. Need at least 300."
        )
    
    if N != y_row:
        raise ValueError(
            f"Dimension mismatch. The number of outputs ({y_row}) should match"
            f" number of samples ({N})"
        )
    
    if max_order not in (1, 2, 3):
        raise ValueError(
            f'Field "max_order" of options should be an integer with values of'
            f" 1, 2 or 3, got {max_order}"
        )

    # Important next check for max_order - as max_order relates to d
    if (d == 2) and (max_order > 2):
        raise ValueError(
            'SALib-HDMR ERRROR: Field "max_order" of options has to be 2 as'
            " d = 2 (X has two columns)"
        )

    if poly_order not in np.arange(1, 11):
        raise ValueError('Field "k" of options should be an integer between 1 to 10.')

    if bootstrap not in np.arange(1, 101):
        raise ValueError(
            'Field "bootstrap" of options should be an integer between 1 to 100.'
        )

    if subset is None:
        subset = y_row // 2
    elif subset not in np.arange(300, N + 1):
        raise ValueError(
            f'Field "subset" of options should be an integer between 300 and {N},'
            f" the number of rows matrix X."
        )

    if (bootstrap == 1) and (subset != y_row):
        subset = y_row

    if alpha < 0.5 or alpha > 1.0:
        raise ValueError(
            'Field "alpha" of options should be a float between 0.5 to 1.0'
        )
    

def _core_params(N, d, poly_order, max_order, bootstrap, subset, extended_base) -> namedtuple:
    """ Core Parameters of HDMR expansion. Please see detailed explanation below

    nc1: Number of component functions in 1st order
    nc2: Number of component functions in 2nd order
    nc3: Number of component functions in 3rd order
    nc_t: Total number of component functions
    nt1: Total number of terms(columns) for a given 1st order component function
    nt2: Total number of terms(columns) for a given 2nd order component function
    nt3: Total number of terms(columns) for a given 3rd order component function
    tnt1: Total number of terms(columns) for all 1st order component functions
    tnt2: Total number of terms(columns) for all 2nd order component functions
    tnt3: Total number of terms(columns) for all 3rd order component functions
    a_tnt: All terms (columns) in a hdmr expansion """

    cp = defaultdict(int)
    cp['n_comp_func'], cp['n_coeff'] = [0] * 3, [0] * 3
    cp['n_comp_func'][0] = d
    cp['n_coeff'][0] = poly_order
    
    if max_order > 1:
        cp['n_comp_func'][1] = math.comb(d, 2)
        cp['n_coeff'][1] = poly_order**2
        if extended_base:
            cp['n_coeff'][1] += 2*poly_order 
        

    if max_order == 3:
        cp['n_comp_func'][2] = math.comb(d, 3)
        cp['n_coeff'][2] = poly_order**3
        if extended_base:
            cp['n_coeff'][2] += 3*poly_order + 3*poly_order**2

    # Setup Bootstrap (if bootstrap > 1)
    idx = np.arange(0, N) if bootstrap == 1 else np.argsort(np.random.rand(N, bootstrap), axis=0)[:subset]
 
    CoreParams = namedtuple('CoreParams', ['N', 'd', 'p_o', 'nc1', 'nc2', 'nc3', 'nc_t', 'nt1', 'nt2', 'nt3', 
                           'tnt1', 'tnt2', 'tnt3', 'a_tnt', 'idx', 'beta', 
                           'gamma', 'S', 'Sa', 'Sb', 'ST'])
    
    hdmr = CoreParams(
        N,
        d,
        poly_order,
        cp['n_comp_func'][0],
        cp['n_comp_func'][1],
        cp['n_comp_func'][2],
        sum(cp['n_comp_func']),
        cp['n_coeff'][0],
        cp['n_coeff'][1],
        cp['n_coeff'][2],
        cp['n_coeff'][0] * cp['n_comp_func'][0],
        cp['n_coeff'][1] * cp['n_comp_func'][1],
        cp['n_coeff'][2] * cp['n_comp_func'][2],
        cp['n_coeff'][0] * cp['n_comp_func'][0] + \
        cp['n_coeff'][1] * cp['n_comp_func'][1] + \
        cp['n_coeff'][2] * cp['n_comp_func'][2],
        idx,
        np.asarray(list(comb(np.arange(0, d), 2))),
        np.asarray(list(comb(np.arange(0, d), 3))),
        np.empty((sum(cp['n_comp_func']), bootstrap)),
        np.empty((sum(cp['n_comp_func']), bootstrap)),
        np.empty((sum(cp['n_comp_func']), bootstrap)),
        np.empty((sum(cp['n_comp_func']), bootstrap))
    )

    return hdmr