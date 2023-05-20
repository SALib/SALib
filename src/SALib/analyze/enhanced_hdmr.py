import math
import warnings
import numpy as np
from pandas import DataFrame as df
from typing import Dict
from types import MethodType
from numpy.linalg import det, pinv, matrix_rank
from scipy.linalg import svd, LinAlgError, solve
from scipy import stats, special
from itertools import combinations as comb, product
from collections import defaultdict, namedtuple

from ..util import read_param_file, ResultDict


def analyze(
    problem: Dict,
    X: np.ndarray,
    Y: np.ndarray,
    max_order: int = 2,
    poly_order: int = 2,
    bootstrap: int = 20,
    subset: int = 500,
    max_iter: int = 100,
    lambdax: float = 0.01,
    alpha: float = 0.95,
    extended_base: bool = True,
    print_to_console: bool = False,
    return_emulator: bool = False,
    seed: int = None,
) -> Dict:
    # Random Seed
    if seed:
        np.random.seed(seed)

    # Check arguments
    Y, problem, subset, max_iter = _check_args(
        problem,
        X,
        Y,
        max_order,
        poly_order,
        bootstrap,
        subset,
        max_iter,
        lambdax,
        alpha,
        extended_base,
    )
    # Instantiate Core Parameters
    hdmr = _core_params(
        *X.shape, np.mean(Y), poly_order, max_order, bootstrap, subset, extended_base
    )
    # Calculate HDMR Basis Matrix
    b_m = _basis_matrix(X, hdmr)
    # Functional ANOVA decomposition
    _fanova(b_m, hdmr, Y, bootstrap, max_iter, lambdax, alpha)
    # HDMR finalize
    Si = _finalize(problem, hdmr, alpha, return_emulator)
    # Print results to console
    if print_to_console:
        _print(Si)

    return Si


def _check_args(
    problem,
    X,
    Y,
    max_order,
    poly_order,
    bootstrap,
    subset,
    max_iter,
    lambdax,
    alpha,
    extended_base,
):
    """Validates all parameters to ensure that they are within the limits"""
    # Make sure Y, output array, is a matrix
    Y = Y.reshape(-1, 1)

    # Get dimensions of input-output
    N, d = X.shape
    y_row = Y.shape[0]

    # If parameter names are not defined
    if not "names" in problem:
        problem["names"] = [f"x_{i}" for i in range(d)]

    # If parameter bounds are not defined
    if not "bounds" in problem:
        problem["bounds"] = [[X[:, i].min(), X[:, i].max()] for i in range(d)]

    # If parameter num_vars are not defined
    if not "num_vars" in problem:
        problem["num_vars"] = d

    # If the length of 'num_vars' in ProblemSpec != Columns in X matrix
    if "num_vars" in problem and problem["num_vars"] != d:
        raise ValueError(
            "SALib-HDMR Error: Problem definition must be consistent with the number of dimension in matrix X"
        )

    # If the length of 'names' in ProblemSpec != Columns in X matrix
    if "names" in problem and len(problem["names"]) != d:
        raise ValueError(
            "SALib-HDMR Error: Problem definition must be consistent with the number of dimension in matrix X"
        )

    # If the length of 'bounds' in ProblemSpec != Columns in X matrix
    if "bounds" in problem and len(problem["bounds"]) != d:
        raise ValueError(
            "SALib-HDMR Error: Problem definition must be consistent with the number of dimension in matrix X"
        )

    # Now check input-output mismatch
    if d == 1:
        raise RuntimeError(
            "SALib-HDMR Error: Matrix X contains only a single column: No point to do"
            " sensitivity analysis when d = 1."
        )

    if N < 300:
        raise RuntimeError(
            f"SALib-HDMR Error: Number of samples in the input matrix X, {N}, is insufficient. Need at least 300."
        )

    if N != y_row:
        raise ValueError(
            f"SALib-HDMR Error: Dimension mismatch. The number of outputs ({y_row}) should match"
            f" number of samples ({N})"
        )

    if max_order not in (1, 2, 3):
        raise ValueError(
            f"SALib-HDMR Error: 'max_order' key of options should be an integer with values of"
            f" 1, 2 or 3, got {max_order}"
        )

    # Important next check for max_order - as max_order relates to d
    if (d == 2) and (max_order > 2):
        max_order = 2
        warnings.warn(
            "SALib-HDMR Warning: max_order is set to 2 due to lack of third input factor"
        )

    if poly_order not in np.arange(1, 11):
        raise ValueError(
            "SALib-HDMR Error: 'poly_order' key of options should be an integer between 1 to 10."
        )

    if bootstrap not in np.arange(1, 101):
        raise ValueError(
            "SALib-HDMR Error: 'bootstrap' key of options should be an integer between 1 to 100."
        )

    if (bootstrap == 1) and (subset != y_row):
        subset = y_row

    if subset is None:
        subset = y_row // 2
    elif subset not in np.arange(300, N + 1):
        raise ValueError(
            f"SALib-HDMR Error: 'subset' key of options should be an integer between 300 and {N}, "
            f"the number of rows matrix X."
        )

    if alpha < 0.5 or alpha > 1.0:
        raise ValueError(
            "SALib-HDMR Error: 'alpha' key of options should be a float between 0.5 to 1.0"
        )

    if extended_base:
        max_iter = None
    else:
        if max_iter not in np.arange(100, 1000):
            raise ValueError(
                "SALib-HDMR Error: 'max_iter' key of options should be between 100 and 1000"
            )

    if lambdax < 0.0 or lambdax > 10:
        raise ValueError(
            "SALib-HDMR Error: 'lambdax' key of options should be in between 0 and 10"
        )

    return Y, problem, subset, max_iter


def _core_params(
    N: int,
    d: int,
    f0: float,
    poly_order: int,
    max_order: int,
    bootstrap: int,
    subset: int,
    extended_base: bool,
) -> namedtuple:
    """This function establishes core parameters of HDMR expansion in a namedtuple datatype.
    These parameters are being used across all functions and procedures.

    Parameters
    ----------
    N : int
        Number of samples in input matrix `X`.
    d : int
        Dimensionality of the problem.
    f0 : float
        Zero-th component function
    poly_order : int
        Polynomial order to be used to calculate orthonormal polynomial.
    max_order : int
        Maximum functional ANOVA expansion order.
    bootstrap : int
        Number of iteration to be used in bootstrap.
    subset : int
        Number of samples to be used in bootstrap.
    extended_base : bool
        Whether to use extended basis matrix or not.

    Returns
    -------
    hdmr : namedtuple
        A variable that hold core parameters of hdmr expansion

    Attributes
    ----------
    N : int
        Number of samples in input matrix `X`.
    d : int
        Dimensionality of the problem.
    max_order : int
        Maximum functional ANOVA expansion order.
    ext_base : bool
        Whether to use extended basis matrix or not.
    subset : int
        Number of samples to be used in bootstrap.
    p_o : int
        Polynomial order to be used to calculated orthonormal polynomial.
    nc1 : int
        Number of component functions in 1st order.
    nc2 : int
        Number of component functions in 2nd order.
    nc3 int
        Number of component functions in 3rd order.
    nc_t : int
        Total number of component functions.
    nt1 : int
        Total number of terms(columns) for a given 1st order component function
    nt2 : int
        Total number of terms(columns) for a given 2nd order component function
    nt3 : int
        Total number of terms(columns) for a given 3rd order component function
    tnt1 : int
        Total number of terms(columns) for all 1st order component functions
    tnt2 : int
        Total number of terms(columns) for all 2nd order component functions
    tnt3 : int
        Total number of terms(columns) for all 3rd order component functions
    a_tnt : int
        All terms (columns) in a hdmr expansion
    x : numpy.array
        Solution of hdmr expansion
    idx : numpy.array
        Indexes of subsamples to be used for bootstrap
    S : numpy.array
        Sensitivity indexes of component functions
    Sa : numpy.array
        Sensitivity indexes of component functions (uncorrelated contribution)
    Sb : numpy.array
        Sensitivity indexes of component functions (correlated contribution)
    ST : numpy.array
        Total Sensitivity indexes of features/inputs
    signf : numpy.array
        Signigicance level for each bootstrap iteration
    beta : numpy.array
        Arrangement of second-order component functions
    gamma : numpy.array
        Arrangement of third-order component functions
    """

    cp = defaultdict(int)
    cp["n_comp_func"], cp["n_coeff"] = [0] * 3, [0] * 3
    cp["n_comp_func"][0] = d
    cp["n_coeff"][0] = poly_order

    if max_order > 1:
        cp["n_comp_func"][1] = math.comb(d, 2)
        cp["n_coeff"][1] = poly_order**2
        if extended_base:
            cp["n_coeff"][1] += 2 * poly_order

    if max_order == 3:
        cp["n_comp_func"][2] = math.comb(d, 3)
        cp["n_coeff"][2] = poly_order**3
        if extended_base:
            cp["n_coeff"][2] += 3 * poly_order + 3 * poly_order**2

    # Setup Bootstrap (if bootstrap > 1)
    idx = (
        np.arange(0, N).reshape(-1, 1)
        if bootstrap == 1
        else np.argsort(np.random.rand(N, bootstrap), axis=0)[:subset]
    )

    CoreParams = namedtuple(
        "CoreParams",
        [
            "N",
            "d",
            "max_order",
            "ext_base",
            "subset",
            "p_o",
            "nc1",
            "nc2",
            "nc3",
            "nc_t",
            "nt1",
            "nt2",
            "nt3",
            "tnt1",
            "tnt2",
            "tnt3",
            "a_tnt",
            "x",
            "idx",
            "S",
            "Sa",
            "Sb",
            "ST",
            "signf",
            "beta",
            "gamma",
            "f0",
        ],
    )

    n_comp_func = cp["n_comp_func"]
    n_coeff = cp["n_coeff"]
    hdmr = CoreParams(
        N,
        d,
        max_order,
        extended_base,
        subset,
        poly_order,
        n_comp_func[0],
        n_comp_func[1],
        n_comp_func[2],
        sum(n_comp_func),
        n_coeff[0],
        n_coeff[1],
        n_coeff[2],
        n_coeff[0] * n_comp_func[0],
        n_coeff[1] * n_comp_func[1],
        n_coeff[2] * n_comp_func[2],
        n_coeff[0] * n_comp_func[0]
        + n_coeff[1] * n_comp_func[1]
        + n_coeff[2] * n_comp_func[2],
        np.zeros(
            n_coeff[0] * n_comp_func[0]
            + n_coeff[1] * n_comp_func[1]
            + n_coeff[2] * n_comp_func[2]
        ),
        idx,
        np.zeros((sum(n_comp_func), bootstrap)),
        np.zeros((sum(n_comp_func), bootstrap)),
        np.zeros((sum(n_comp_func), bootstrap)),
        np.zeros((sum(n_comp_func), bootstrap)),
        np.zeros((sum(n_comp_func), bootstrap)),
        np.array(list(comb(range(d), 2))),
        np.array(list(comb(range(d), 3))),  # Returns empty list when d < 3
        f0,
    )

    return hdmr


def _basis_matrix(X, hdmr):
    """Basis matrix represents the base of the component functions.
    It is built with orthonormal polynomials for each inputs so that
    it represents the data at its best. Linear combination of columns
    of this matrix forms the component functions.

    Parameters
    ----------
    X : numpy.array
        Model input matrix
    hdmr : namedtuple
        Core parameters of hdmr expansion

    Returns
    -------
    b_m : numpy.array
        Basis matrix
    """
    # Compute normalized X-values
    X_n = (X - np.tile(X.min(0), (hdmr.N, 1))) / np.tile(
        (X.max(0)) - X.min(0), (hdmr.N, 1)
    )

    # Compute Orthonormal Polynomial Coefficients
    coeff = _orth_poly_coeff(X_n, hdmr)

    # Initialize Basis Matrix
    b_m = np.zeros((hdmr.N, hdmr.a_tnt))

    # First order columns of basis matrix
    col = 0
    for i, j in product(range(hdmr.d), range(hdmr.p_o)):
        b_m[:, col] = np.polyval(coeff[j, : j + 2, i], X_n[:, i])
        col += 1

    # Second order columns of basis matrix
    if hdmr.max_order > 1:
        for i in _prod(range(0, hdmr.d - 1), range(1, hdmr.d)):
            if hdmr.ext_base:
                b_m[:, col : col + hdmr.p_o] = b_m[
                    :, i[0] * hdmr.p_o : (i[0] + 1) * hdmr.p_o
                ]
                col += hdmr.p_o
                b_m[:, col : col + hdmr.p_o] = b_m[
                    :, i[1] * hdmr.p_o : (i[1] + 1) * hdmr.p_o
                ]
                col += hdmr.p_o

            for j in _prod(
                range(i[0] * hdmr.p_o, (i[0] + 1) * hdmr.p_o),
                range(i[1] * hdmr.p_o, (i[1] + 1) * hdmr.p_o),
            ):
                b_m[:, col] = np.multiply(b_m[:, j[0]], b_m[:, j[1]])
                col += 1

    # Third order columns of basis matrix
    if hdmr.max_order == 3:
        for i in _prod(range(0, hdmr.d - 2), range(1, hdmr.d - 1), range(2, hdmr.d)):
            if hdmr.ext_base:
                b_m[:, col : col + hdmr.p_o] = b_m[
                    :, i[0] * hdmr.p_o : (i[0] + 1) * hdmr.p_o
                ]
                col += hdmr.p_o
                b_m[:, col : col + hdmr.p_o] = b_m[
                    :, i[1] * hdmr.p_o : (i[1] + 1) * hdmr.p_o
                ]
                col += hdmr.p_o
                b_m[:, col : col + hdmr.p_o] = b_m[
                    :, i[2] * hdmr.p_o : (i[2] + 1) * hdmr.p_o
                ]
                col += hdmr.p_o
                b_m[:, col : col + hdmr.p_o**2] = b_m[
                    :,
                    hdmr.tnt1
                    + (2 * hdmr.nt1) * (i[0] + 1)
                    + i[0] * (hdmr.p_o**2) : hdmr.tnt1
                    + (i[0] + 1) * (hdmr.p_o**2 + 2 * hdmr.nt1),
                ]
                col += hdmr.p_o**2
                b_m[:, col : col + hdmr.p_o**2] = b_m[
                    :,
                    hdmr.tnt1
                    + (2 * hdmr.nt1) * (i[1] + 1)
                    + i[1] * (hdmr.p_o**2) : hdmr.tnt1
                    + (i[1] + 1) * (hdmr.p_o**2 + 2 * hdmr.nt1),
                ]
                col += hdmr.p_o**2
                b_m[:, col : col + hdmr.p_o**2] = b_m[
                    :,
                    hdmr.tnt1
                    + (2 * hdmr.nt1) * (i[2] + 1)
                    + i[2] * (hdmr.p_o**2) : hdmr.tnt1
                    + (i[2] + 1) * (hdmr.p_o**2 + 2 * hdmr.nt1),
                ]
                col += hdmr.p_o**2

            for j in _prod(
                range(i[0] * hdmr.p_o, (i[0] + 1) * hdmr.p_o),
                range(i[1] * hdmr.p_o, (i[1] + 1) * hdmr.p_o),
                range(i[2] * hdmr.p_o, (i[2] + 1) * hdmr.p_o),
            ):
                b_m[:, col] = np.multiply(
                    np.multiply(b_m[:, j[0]], b_m[:, j[1]]), b_m[:, j[2]]
                )
                col += 1

    return b_m


def _orth_poly_coeff(X, hdmr):
    """Calculated orthonormal polynomial coefficients based on a given input matrix `X`

    Parameters
    ----------
    X : numpy.array
        Input matrix `X`
    hdmr : namedtuple
        Core parameters of hdmr expansion

    Returns
    -------
    coeff : numpy.array
        Orthonormal polynomial coefficients from highes degree to constant term
          with trailing zeros

    Notes
    ----------
    Please see the reference below

    .. [1] Szegő, G. 1975. . Orthogonal Polynomials. American Mathematical Society.
    """
    p_o_1 = hdmr.p_o + 1
    M = np.zeros((p_o_1, p_o_1, hdmr.d))
    for i in range(hdmr.d):
        k = 0
        for j in range(p_o_1):
            for z in range(p_o_1):
                M[j, z, i] = sum(X[:, i] ** k) / hdmr.N
                k += 1
            k = j + 1

    coeff = np.zeros((hdmr.p_o, p_o_1, hdmr.d))
    for i, j in product(range(hdmr.d), range(hdmr.p_o)):
        z = range(j + 2)
        for k in z:
            z__k = list(z)
            z__k.pop(k)
            det_ij = det(M[: j + 1, : j + 1, i]) * det(M[: j + 2, : j + 2, i])
            coeff[j, j + 1 - k, i] = (
                (-1) ** (j + k + 1) * det(M[: j + 1, z__k, i]) / np.sqrt(det_ij)
            )

    return coeff


def _prod(*args):
    """Generator that returns unique cartesian product of given tuple arguments

    Parameters
    ----------
    *args : List[Tuple]
        Variable length argument list.

    Yields
    ------
    Tuple
        A non-duplicate numbers in a tuple.
    """
    seen = set()
    for prod in product(*args):
        prod_set = frozenset(prod)
        if len(prod_set) != len(prod):
            continue
        if prod_set not in seen:
            seen.add(prod_set)
            yield prod


def _fanova(b_m, hdmr, Y, bootstrap, max_iter, lambdax, alpha):
    """Functional ANOVA decomposition provides two main approach namely
    extended base approach and non-extended base approach which follow the guidelines
    from [1] and [2]. Extended base in this manner is to provide additional information
    to guarantee hierarchical orthogonality.

    Parameters
    ----------
    b_m : numpy.array
        Basis matrix
    hdmr : namedtuple
        Core parameters of hdmr expansion
    Y : numpy.array
        Model output
    bootstrap : int
        Number of iteration to be used in bootstrap
    max_iter : int
        Maximum number of iteration used in backfitting algorithm
    lambdax : float
        Penalty term for ridge regression
    alpha : float
        Significant level

    Notes
    -----
    .. [1] Li, G., Rabitz, H., Yelvington, P., Oluwole, O., Bacon, F., Kolb, C., and Schoendorf, J. 2010.
        Global Sensitivity Analysis for Systems with Independent and/or Correlated Inputs.
        The Journal of Physical Chemistry A, 114(19), p.6022-6032.
    .. [2] Li, G., Rabitz, H. General formulation of HDMR component functions with independent and correlated variables.
        J Math Chem 50, 99–130 (2012). https://doi.org/10.1007/s10910-011-9898-0
    """
    for t in range(bootstrap):
        # Extract model output for a corresponding bootstrap iteration
        Y_idx = Y[hdmr.idx[:, t], 0]
        # Subtract mean from it
        Y_idx -= np.mean(Y_idx)

        if hdmr.ext_base:
            cost = _cost_matrix(b_m[hdmr.idx[:, t], :], hdmr)
            _d_morph(b_m[hdmr.idx[:, t], :], cost, Y_idx, bootstrap, hdmr)
        else:
            Y_res = _first_order(
                b_m[hdmr.idx[:, t], : hdmr.tnt1], Y_idx, max_iter, lambdax, hdmr
            )
            if hdmr.max_order > 1:
                Y_res = _second_order(
                    b_m[hdmr.idx[:, t], hdmr.tnt1 : hdmr.tnt1 + hdmr.tnt2],
                    Y_res,
                    max_iter,
                    lambdax,
                    hdmr,
                )
            if hdmr.max_order == 3:
                _third_order(
                    b_m[hdmr.idx[:, t], hdmr.tnt1 + hdmr.tnt2 :], Y_res, lambdax, hdmr
                )

        # Calculate component functions
        Y_e = _comp_func(b_m[hdmr.idx[:, t], :], hdmr)
        # Test significancy
        _f_test(Y_idx, Y_e, alpha, hdmr, t)
        # Sensitivity Analysis
        _ancova(Y_idx, Y_e, hdmr, t)

    return hdmr


def _cost_matrix(b_m, hdmr):
    """Cost matrix holds the information about hierarchical orthogonality.
    This matrix is arranged in such a way that it satisfies component functions
    that hiearchical to each other are orthogonal.

    Parameters
    ----------
    b_m : numpy.array
        Basis matrix
    hdmr : namedtuple
        Core parameters of hdmr expansion

    Returns
    -------
    cost : numpy.array
        Cost matrix
    """
    cost = np.zeros((hdmr.a_tnt, hdmr.a_tnt))

    range_2nd_1 = lambda x: range(
        hdmr.tnt1 + (x) * hdmr.nt2, hdmr.tnt1 + (x + 1) * hdmr.nt2
    )
    range_2nd_2 = lambda x: range(
        hdmr.tnt1 + (x) * hdmr.nt2, hdmr.tnt1 + (x) * hdmr.nt2 + hdmr.p_o * 2
    )
    range_3rd_1 = lambda x: range(
        hdmr.tnt1 + hdmr.tnt2 + (x) * hdmr.nt3,
        hdmr.tnt1 + hdmr.tnt2 + (x + 1) * hdmr.nt3,
    )
    range_3rd_2 = lambda x: range(
        hdmr.tnt1 + hdmr.tnt2 + (x) * hdmr.nt3,
        hdmr.tnt1 + hdmr.tnt2 + (x) * hdmr.nt3 + 3 * hdmr.p_o + 3 * hdmr.p_o**2,
    )

    if hdmr.max_order > 1:
        sr_i = np.mean(b_m, axis=0, keepdims=True)
        sr_ij = np.zeros((2 * hdmr.p_o + 1, hdmr.nt2))
        ct = 0
        for _ in _prod(range(0, hdmr.d - 1), range(1, hdmr.d)):
            sr_ij[0, :] = sr_i[0, range_2nd_1(ct)]
            sr_ij[1:, :] = (
                b_m[:, range_2nd_2(ct)].T @ b_m[:, range_2nd_1(ct)]
            ) / hdmr.subset
            cost[np.ix_(range_2nd_1(ct), range_2nd_1(ct))] = sr_ij.T @ sr_ij
            ct += 1
    if hdmr.max_order == 3:
        sr_ijk = np.zeros((3 * hdmr.p_o + 3 * hdmr.p_o**2 + 1, hdmr.nt3))
        ct = 0
        for _ in _prod(range(0, hdmr.d - 2), range(1, hdmr.d - 1), range(2, hdmr.d)):
            sr_ijk[0, :] = sr_i[0, range_3rd_1(ct)]
            sr_ijk[1:, :] = (
                b_m[:, range_3rd_2(ct)].T @ b_m[:, range_3rd_1(ct)] / hdmr.subset
            )
            cost[np.ix_(range_3rd_1(ct), range_3rd_1(ct))] = sr_ijk.T @ sr_ijk
            ct += 1

    return cost


def _d_morph(b_m, cost, Y_idx, subset, hdmr):
    """D-Morph Regression finds the best solution that aligns with the cost
    matrix. Cost matrix in this case represents the hierarchical orthogonality
    between component functions.

    Parameters
    ----------
    b_m : numpy.array
        Basis matrix for all component functions
    cost : numpy.array
        Cost matrix that satisfies hierarchical orthogonality
    Y_idx : numpy.array
        Model output for a single bootstrap iteration
    subset : int
        Number of subsamples
    hdmr : namedtuple
        Core parameters of hdmr expansion

    Notes
    -----
    Detailed information about D-Morph Regression can be found at
    .. [1] Li, G., Rey-de-Castro, R. & Rabitz, H. D-MORPH regression for modeling
        with fewer unknown parameters than observation data.
        J Math Chem 50, 1747–1764 (2012). https://doi.org/10.1007/s10910-012-0004-z
    """
    # Left Matrix Multiplication with the transpose of cost matrix
    a = (b_m.T @ b_m) / subset  # LHS
    b = (b_m.T @ Y_idx) / subset  # RHS
    try:
        # Pseudo-Inverse of LHS
        a_pinv = pinv(a, hermitian=True)
        rank = matrix_rank(a)
        # Least-Square Solution
        x = a_pinv @ b
        # Projection Matrix
        pr = np.eye(hdmr.a_tnt) - (a_pinv @ a)
        pb = pr @ cost
        U, _, Vh = svd(pb)
    except LinAlgError:
        print("Pseudo-Inverse did not converge")
    
    nullity = min(b_m.shape) - rank
    V = Vh.T
    U = np.delete(U, range(0, nullity), axis=1)
    V = np.delete(V, range(0, nullity), axis=1)

    # D-Morph Regression Solution
    hdmr.x[:] = V @ pinv(U.T @ V) @ U.T @ x


def _first_order(b_m1, Y_idx, max_iter, lambdax, hdmr):
    """Sequential determination of first order component functions.
    First, it computes component functions via ridge regression, i.e.
    fitting model inputs/features to the model output. Later, it takes
    advantage of backfitting algorithm to satisfy hierarchical orthogonality.
    Backfitting algorithm does not guarantee the hierarchical orthogonality.
    We suggest to set extended base option to `True` for those who want to
    guaranteed functional ANOVA expansion.

    Parameters
    ----------
    b_m1 : numpy.array
        Basis matrix for first-order component functions
    Y_idx : numpy.array
        Model output for a single bootstrap iteration
    max_iter : int
        Maximum number of iteration used in backfitting algorithm
    lambdax : float
        Penalty term for ridge regression
    hdmr : namedtuple
        Core parameters of hdmr expansion

    Returns
    -------
    Y_res : numpy.array
        Residual model output
    """
    # Temporary first order component matrix
    Y_i = np.empty((hdmr.subset, hdmr.nc1))
    # Initialize iter
    iter = 0
    # To increase readibility
    n1 = hdmr.nt1
    # L2 Penalty
    lambda_eye = lambdax * np.identity(n1)
    for i in range(hdmr.nc1):
        try:
            # Left hand side
            a = (
                b_m1[:, i * n1 : n1 * (i + 1)].T @ b_m1[:, i * n1 : n1 * (i + 1)]
            ) / hdmr.subset
            # Adding L2 Penalty (Ridge Regression)
            a += lambda_eye
            # Right hand side
            b = (b_m1[:, i * n1 : n1 * (i + 1)].T @ Y_idx) / hdmr.subset
            # Solution
            hdmr.x[i * n1 : n1 * (i + 1)] = solve(a, b)
            # Component functions
            Y_i[:, i] = b_m1[:, i * n1 : n1 * (i + 1)] @ hdmr.x[i * n1 : n1 * (i + 1)]
        except LinAlgError:
            print(
                "Least-square regression did not converge. Try increasing L2 penalty term."
            )

    # Backfitting method
    var_old = np.square(hdmr.x[: hdmr.tnt1])
    z_t = list(range(hdmr.d))
    while not ((var_max < 1e-4) or (iter > max_iter)):
        for i in range(hdmr.d):
            z = z_t[:]
            z.remove(i)
            Y_res = Y_idx - np.sum(Y_i[:, z], axis=1)
            # Left hand side
            a = (
                b_m1[:, i * n1 : n1 * (i + 1)].T @ b_m1[:, i * n1 : n1 * (i + 1)]
            ) / hdmr.subset
            # Right hand side
            b = (b_m1[:, i * n1 : n1 * (i + 1)].T @ Y_res) / hdmr.subset
            # Solution
            hdmr.x[i * n1 : n1 * (i + 1)] = solve(a, b)
            # Component functions
            Y_i[:, i] = b_m1[:, i * n1 : n1 * (i + 1)] @ hdmr.x[i * n1 : n1 * (i + 1)]

        var_max = np.absolute(var_old - np.square(hdmr.x[: hdmr.tnt1])).max()
        iter += 1

        var_old = np.square(hdmr.x[: hdmr.tnt1])

    return Y_idx - np.sum(Y_i, axis=1)


def _second_order(b_m2, Y_res, max_iter, lambdax, hdmr):
    """Sequential determination of second-order component functions.
    First, it computes component functions via ridge regression, i.e.
    fitting model inputs/features to the model output. Later, it takes
    advantage of backfitting algorithm to satisfy hierarchical orthogonality.
    Backfitting algorithm does not guarantee the hierarchical orthogonality.
    We suggest to set extended base option to `True` for those who want to
    guaranteed functional ANOVA expansion.

    Parameters
    ----------
    b_m2 : numpy.array
        Basis matrix for second-order component functions
    Y_res : numpy.array
        Residual model output
    max_iter : int
        Maximum number of iteration used in backfitting algorithm
    lambdax : float
        Penalty term for ridge regression
    hdmr : namedtuple
        Core parameters of hdmr expansion

    Returns
    -------
    Y_res : numpy.array
        Residual model output
    """
    # Temporary second order component matrix
    Y_ij = np.empty((hdmr.subset, hdmr.nc2))
    # To increase readibility
    n2 = hdmr.nt2
    # Initialize iteration counter
    iter = 0
    # L2 Penalty
    lambda_eye = lambdax * np.identity(n2)
    for i in range(hdmr.nc2):
        try:
            # Left hand side
            a = (
                b_m2[:, i * n2 : n2 * (i + 1)].T @ b_m2[:, i * n2 : n2 * (i + 1)]
            ) / hdmr.subset
            # Adding L2 Penalty (Ridge Regression)
            a += lambda_eye
            # Right hand side
            b = (b_m2[:, i * n2 : n2 * (i + 1)].T @ Y_res) / hdmr.subset
            # Solution
            hdmr.x[hdmr.tnt1 + i * n2 : hdmr.tnt1 + n2 * (i + 1)] = solve(a, b)
            # Component functions
            Y_ij[:, i] = (
                b_m2[:, i * n2 : n2 * (i + 1)]
                @ hdmr.x[hdmr.tnt1 + i * n2 : hdmr.tnt1 + n2 * (i + 1)]
            )
        except LinAlgError:
            print(
                "Least-square regression did not converge. Please increase L2 penalty term!"
            )

    var_old = np.square(hdmr.x[hdmr.tnt1 : hdmr.tnt1 + hdmr.tnt2])
    # Backfitting method
    while True:
        for i in range(hdmr.nc2):
            z = list(range(hdmr.nc2))
            z.remove(i)
            Y_r = Y_res - np.sum(Y_ij[:, z], axis=1)
            # Left hand side
            a = (
                b_m2[:, i * n2 : n2 * (i + 1)].T @ b_m2[:, i * n2 : n2 * (i + 1)]
            ) / hdmr.subset
            # Right hand side
            b = (b_m2[:, i * n2 : n2 * (i + 1)].T @ Y_r) / hdmr.subset
            # Solution
            hdmr.x[hdmr.tnt1 + i * n2 : hdmr.tnt1 + n2 * (i + 1)] = solve(a, b)
            # Component functions
            Y_ij[:, i] = (
                b_m2[:, i * n2 : n2 * (i + 1)]
                @ hdmr.x[hdmr.tnt1 + i * n2 : hdmr.tnt1 + n2 * (i + 1)]
            )

        var_max = np.absolute(
            var_old - np.square(hdmr.x[hdmr.tnt1 : hdmr.tnt1 + hdmr.tnt2])
        ).max()
        iter += 1

        if (var_max < 1e-4) or (iter > max_iter):
            break

        var_old = np.square(hdmr.x[hdmr.tnt1 : hdmr.tnt1 + hdmr.tnt2])

    return Y_res - np.sum(Y_ij, axis=1)


def _third_order(b_m3, Y_res, lambdax, hdmr):
    """Sequential determination of third-order component functions.
    it computes component functions via ridge regression, i.e.
    fitting model inputs/features to the model output.

    Parameters
    ----------
    b_m3 : numpy.array
        Basis matrix for third-order component functions
    Y_res : numpy.array
        Residual model output
    lambdax : float
        Penalty term for ridge regression
    hdmr : namedtuple
        Core parameters of hdmr expansion

    Notes
    -----
    Backfitting algorithm is not used here because the residual model
    output, `Y_res`, might not be converged to arrays of zero.
    """
    # To increase readibility
    n3 = hdmr.nt3
    # L2 Penalty
    lambda_eye = lambdax * np.identity(n3)
    for i in range(hdmr.nc3):
        try:
            # Left hand side
            a = (
                b_m3[:, i * n3 : n3 * (i + 1)].T @ b_m3[:, i * n3 : n3 * (i + 1)]
            ) / hdmr.subset
            # Adding L2 Penalty (Ridge Regression)
            a += lambda_eye
            # Right hand side
            b = (b_m3[:, i * n3 : n3 * (i + 1)].T @ Y_res) / hdmr.subset
            # Solution
            hdmr.x[
                hdmr.tnt1 + hdmr.tnt2 + i * n3 : hdmr.tnt1 + hdmr.tnt2 + n3 * (i + 1)
            ] = solve(a, b)
        except LinAlgError:
            print(
                "Least-square regression did not converge. Please increase L2 penalty term!"
            )


def _comp_func(b_m, hdmr):
    """Computes the component function based on basis matrix and the solution

    Parameters
    ----------
    b_m : numpy.array
        Basis matrix
    hdmr : namedtuple
        Core parameters of hdmr expansion

    Returns
    -------
    Y_e : numpy.array
        Emualated model output for each components
    """
    Y_t = np.zeros((hdmr.subset, hdmr.a_tnt))
    Y_e = np.zeros((hdmr.subset, hdmr.nc_t))

    # Temporary matrix
    Y_t = np.multiply(b_m, np.tile(hdmr.x, [hdmr.subset, 1]))

    # First order component functions
    for i in range(hdmr.nc1):
        Y_e[:, i] = np.sum(Y_t[:, i * hdmr.p_o : (i + 1) * hdmr.p_o], axis=1)

    # Second order component functions
    if hdmr.max_order > 1:
        for i in range(hdmr.nc2):
            Y_e[:, hdmr.nc1 + i] = np.sum(
                Y_t[:, hdmr.tnt1 + (i) * hdmr.nt2 : hdmr.tnt1 + (i + 1) * hdmr.nt2],
                axis=1,
            )

    # Third order component functions
    if hdmr.max_order == 3:
        for i in range(hdmr.nc3):
            Y_e[:, hdmr.nc1 + hdmr.nc2 + i] = np.sum(
                Y_t[
                    :,
                    hdmr.tnt1
                    + hdmr.tnt2
                    + (i) * hdmr.nt3 : hdmr.tnt1
                    + hdmr.tnt2
                    + (i + 1) * hdmr.nt3,
                ],
                axis=1,
            )

    return Y_e


def _ancova(Y_idx, Y_e, hdmr, t):
    """Analysis of Covariance. It calculates uncorrelated and correlated contribution
    to the model output variance

    Parameters
    ----------
    Y_idx : numpy.array
        Model output for a single bootstrap iteration
    Y_e : numpy.array
        Emulated output
    hdmr : namedtuple
        Core parameters of hdmr expansion
    t : int
        Iteration step of bootstrap

    Notes
    -----
    Please see the reference below

    .. [1] Li, G., Rabitz, H., Yelvington, P., Oluwole, O., Bacon, F., Kolb, C., and Schoendorf, J. 2010.
        Global Sensitivity Analysis for Systems with Independent and/or Correlated Inputs.
        The Journal of Physical Chemistry A, 114(19), p.6022-6032.
    """
    # Compute the sum of all Y_em terms
    Y_sum = np.sum(Y_e, axis=1)

    # Total Variance
    tot_v = np.var(Y_idx)

    # Analysis of covariance
    for j in range(hdmr.nc_t):
        # Covariance matrix of jth term of Y_em and actual Y
        c = np.cov(np.stack((Y_e[:, j], Y_idx), axis=0))

        # Total sensitivity of jth term
        hdmr.S[j, t] = c[0, 1] / tot_v  # Eq. 19

        # Covariance matrix of jth term with emulator Y without jth term
        c = np.cov(np.stack((Y_e[:, j], Y_sum - Y_e[:, j]), axis=0))

        # Structural contribution of jth term
        hdmr.Sa[j, t] = c[0, 0] / tot_v  # Eq. 20

        # Correlative contribution of jth term
        hdmr.Sb[j, t] = c[0, 1] / tot_v  # Eq. 21


def _f_test(Y_idx, Y_e, alpha, hdmr, t):
    """Finds which term makes significant contribution to the model output.
    This statistical analysis is done by F-test which uses F-distribution.
    """
    # Sum of squared residuals of bigger model (all params included)
    SSR1 = ((Y_idx - np.sum(Y_e, axis=1)) ** 2).sum()
    # All parameters
    p1 = hdmr.a_tnt
    # Now adding each term to the model
    for i in range(hdmr.nc_t):
        # Model with ith term excluded
        Y_res = Y_idx - Y_e[:, i]
        # Number of parameters of proposed model (order dependent)
        if i < hdmr.nc1:
            p0 = p1 - hdmr.nt1  # 1st order
        elif hdmr.nc1 <= i < (hdmr.nc1 + hdmr.nc2):
            p0 = p1 - hdmr.nt2  # 2nd order
        else:
            p0 = p1 - hdmr.nt3  # 3rd order
        # Sum of squared residuals of smaller model (all params except current terms)
        SSR0 = np.sum(np.square(Y_res))
        # Now calculate the F_stat (F_stat > 0 -> SSR1 < SSR0 )
        F_stat = ((SSR0 - SSR1) / (p1 - p0)) / (SSR1 / (hdmr.subset - p1))
        # Now calculate critical F value
        F_crit = stats.f.ppf(q=alpha, dfn=p1 - p0, dfd=hdmr.subset - p1)
        # Now determine whether to accept ith component into model
        if F_stat > F_crit:
            # ith term is significant and should be included in model
            hdmr.signf[i, t] = 1


def _finalize(problem, hdmr, alpha, return_emulator):
    """Creates ResultDict to be returned."""

    # Create Sensitivity Indices Result Dictionary
    keys = (
        "Sa",
        "Sa_conf",
        "Sb",
        "Sb_conf",
        "S",
        "S_conf",
        "Significant",
        "Sa_sum",
        "Sa_sum_conf",
        "Sb_sum",
        "Sb_sum_conf",
        "S_sum",
        "S_sum_conf",
    )
    Si = ResultDict((k, np.zeros(hdmr.nc_t)) for k in keys)

    Si["Term"] = problem["names"]
    Si["ST"] = np.full(hdmr.nc_t, np.nan)
    Si["ST_conf"] = np.full(hdmr.nc_t, np.nan)

    # Z score
    def z(p):
        return (-1) * np.sqrt(2) * special.erfcinv(p * 2)

    # Multiplier for confidence interval
    mult = z(alpha + (1 - alpha) / 2)

    # Compute the total sensitivity of each parameter/coefficient
    for r in range(hdmr.d):
        if hdmr.max_order == 1:
            TS = hdmr.S[r, :]
        elif hdmr.max_order == 2:
            ij = hdmr.d + np.where(np.sum(hdmr.beta == r, axis=1) == 1)[0]
            TS = np.sum(hdmr.S[np.append(r, ij), :], axis=0)
        elif hdmr.max_order == 3:
            ij = hdmr.d + np.where(np.sum(hdmr.beta == r, axis=1) == 1)[0]
            ijk = hdmr.d + hdmr.nc2 + np.where(np.sum(hdmr.nc3 == r, axis=1) == 1)[0]
            TS = np.sum(hdmr.S[np.append(r, np.append(ij, ijk)), :], axis=0)
        Si["ST"][r] = np.mean(TS)
        Si["ST_conf"][r] = mult * np.std(TS)

    # Generate index column for printing results
    ct = 0

    for i in range(hdmr.d):
        Si["Term"][ct] = problem["names"][i]
        ct += 1

    if hdmr.max_order > 1:
        for i in range(hdmr.nc2):
            Si["Term"][ct] = "/".join(
                [problem["names"][hdmr.beta[i, 0]], problem["names"][hdmr.beta[i, 1]]]
            )
            ct += 1

    if hdmr.max_order == 3:
        for i in range(hdmr.nc3):
            Si["Term"][ct] = "/".join(
                [
                    problem["names"][hdmr.gamma[i, 0]],
                    problem["names"][hdmr.gamma[i, 1]],
                    problem["names"][hdmr.gamma[i, 2]],
                ]
            )
            ct += 1

    # Assign Bootstrap Results to Si Dict
    Si["Sa"] = np.mean(hdmr.Sa, axis=1)
    Si["Sb"] = np.mean(hdmr.Sb, axis=1)
    Si["S"] = np.mean(hdmr.S, axis=1)
    Si["Sa_sum"] = np.mean(np.sum(hdmr.Sa, axis=0))
    Si["Sb_sum"] = np.mean(np.sum(hdmr.Sb, axis=0))
    Si["S_sum"] = np.mean(np.sum(hdmr.S, axis=0))

    # Compute Confidence Interval
    Si["Sa_conf"] = mult * np.std(hdmr.Sa, axis=1)
    Si["Sb_conf"] = mult * np.std(hdmr.Sb, axis=1)
    Si["S_conf"] = mult * np.std(hdmr.S, axis=1)
    Si["Sa_sum_conf"] = mult * np.std(np.sum(hdmr.Sa))
    Si["Sb_sum_conf"] = mult * np.std(np.sum(hdmr.Sb))
    Si["S_sum_conf"] = mult * np.std(np.sum(hdmr.S))

    # F-test # of selection to print out
    Si["select"] = hdmr.signf.flatten()

    # Bind emulator method to the ResultDict
    if return_emulator:
        Si["hdmr"] = hdmr
        Si.emulate = MethodType(emulate, Si)

    # Bind Pandas Dataframe conversion to the ResultDict
    Si.to_df = MethodType(to_df, Si)

    return Si


def to_df(self):
    """Conversion method to Pandas DataFrame. To be attached to ResultDict.

    Returns
    -------
    Pandas DataFrame
    """
    names = self["Term"]

    # Only convert these elements in dict to DF
    include_list = ["Sa", "Sb", "S", "ST"]
    include_list += [f"{name}_conf" for name in include_list]
    new_spec = {k: v for k, v in self.items() if k in include_list}

    return df(new_spec, index=names)


def emulate(self, X):
    """Emulates model output with new input data.

    Constructs orthonormal polynomials with new input matrix, X, and multiplies it with
    solution array, hdmr.x

    Compares emulated results with observed vector, Y.

    Returns
    ========
    Y_em : numpy.array
        Emulated output
    """
    # Calculate HDMR Basis Matrix
    b_m = _basis_matrix(X, self["hdmr"])

    # Take average solution over bootstrap
    self["hdmr"].x = np.mean(self["hdmr"].x, axis=1)

    # Get component functions
    Y_em = _comp_func(b_m, self["hdmr"])

    return np.sum(Y_em, axis=1) + self["hdmr"].f0


def _print(Si):
    d = len(Si["ST"])
    print("\n")
    print(
        "Term    \t      Sa            Sb             S             ST         #select "
    )
    print("-" * 84)  # Header break

    format1 = "%-11s   \t %5.2f (\261%.2f) %5.2f (\261%.2f) %5.2f (\261%.2f) %5.2f (\261%.2f)    %-3.0f"  # noqa: E501
    format2 = "%-11s   \t %5.2f (\261%.2f) %5.2f (\261%.2f) %5.2f (\261%.2f)                  %-3.0f"  # noqa: E501

    for i in range(len(Si["Sa"])):
        if i < d:
            print(
                format1
                % (
                    Si["Term"][i],
                    Si["Sa"][i],
                    Si["Sa_conf"][i],
                    Si["Sb"][i],
                    Si["Sb_conf"][i],
                    Si["S"][i],
                    Si["S_conf"][i],
                    Si["ST"][i],
                    Si["ST_conf"][i],
                    np.sum(Si["select"][i]),
                )
            )
        else:
            print(
                format2
                % (
                    Si["Term"][i],
                    Si["Sa"][i],
                    Si["Sa_conf"][i],
                    Si["Sb"][i],
                    Si["Sb_conf"][i],
                    Si["S"][i],
                    Si["S_conf"][i],
                    np.sum(Si["select"][i]),
                )
            )

    print("-" * 84)  # Header break

    format3 = "%-11s   \t %5.2f (\261%.2f) %5.2f (\261%.2f) %5.2f (\261%.2f)"
    print(
        format3
        % (
            "Sum",
            Si["Sa_sum"],
            Si["Sa_sum_conf"],
            Si["Sb_sum"],
            Si["Sb_sum_conf"],
            Si["S_sum"],
            Si["S_sum_conf"],
        )
    )

    keys = ("Sa_sum", "Sb_sum", "S_sum", "Sa_sum_conf", "Sb_sum_conf", "S_sum_conf")
    for k in keys:
        Si.pop(k, None)
