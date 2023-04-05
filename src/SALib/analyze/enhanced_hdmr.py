from typing import Dict

import numpy as np


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