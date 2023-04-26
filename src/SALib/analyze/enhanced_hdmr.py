import math
import numpy as np
from typing import Dict
from numpy.linalg import det
from scipy.linalg import pinv, svd, LinAlgError, solve
from itertools import combinations as comb, product
from collections import defaultdict, namedtuple





def analyze(
    problem: Dict,
    X: np.ndarray,
    Y: np.ndarray,
    max_order: int = 2,
    poly_order: int = 2,
    bootstrap: int = 20,
    subset: int = None,
    max_iter: int = None,
    lambdax: float = None,
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
    # Calculate HDMR Basis Matrix   
    b_m = _basis_matrix(X, hdmr, max_order, extended_base)
    # Functional ANOVA decomposition
    _fanova(b_m, hdmr, Y, bootstrap, max_order, subset, extended_base, max_iter, lambdax)


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
    idx = np.arange(0, N).reshape(-1, 1) if bootstrap == 1 else np.argsort(np.random.rand(N, bootstrap), axis=0)[:subset]
 
    CoreParams = namedtuple('CoreParams', ['N', 'd', 'p_o', 'nc1', 'nc2', 'nc3', 'nc_t', 'nt1', 'nt2', 'nt3', 
                           'tnt1', 'tnt2', 'tnt3', 'a_tnt', 'x', 'idx', 'beta', 
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
        np.zeros(
            cp['n_coeff'][0] * cp['n_comp_func'][0] + \
            cp['n_coeff'][1] * cp['n_comp_func'][1] + \
            cp['n_coeff'][2] * cp['n_comp_func'][2]
        ),
        idx,
        np.asarray(list(comb(np.arange(0, d), 2))),
        np.asarray(list(comb(np.arange(0, d), 3))),
        np.zeros((sum(cp['n_comp_func']), bootstrap)),
        np.zeros((sum(cp['n_comp_func']), bootstrap)),
        np.zeros((sum(cp['n_comp_func']), bootstrap)),
        np.zeros((sum(cp['n_comp_func']), bootstrap))
    )

    return hdmr


def _basis_matrix(X, hdmr, max_order, extended_base):
    # Compute normalized X-values
    X_n = (X - np.tile(X.min(0), (hdmr.N, 1))) / np.tile((X.max(0)) - X.min(0), (hdmr.N, 1))
    
    # Compute Orthonormal Polynomial Coefficients
    coeff = _orth_poly_coeff(X_n, hdmr)
    # Initialize Basis Matrix
    b_m = np.zeros((hdmr.N, hdmr.a_tnt))
    # Start Column Counter
    col = 0

    # First order columns of basis matrix
    for i in range(hdmr.d):
        for j in range(hdmr.p_o):
            b_m[:, col] = np.polyval(coeff[j, :j+2, i], X_n[:, i])
            col += 1

    # Second order columns of basis matrix
    if max_order > 1:
        for i in _prod(range(0, hdmr.d-1), range(1, hdmr.d)):
            if extended_base:
                b_m[:, col:col+hdmr.p_o] = b_m[:, i[0]*hdmr.p_o:(i[0]+1)*hdmr.p_o]
                col += hdmr.p_o
                b_m[:, col:col+hdmr.p_o] = b_m[:, i[1]*hdmr.p_o:(i[1]+1)*hdmr.p_o]
                col += hdmr.p_o
            
            for j in _prod(range(i[0]*hdmr.p_o, (i[0]+1)*hdmr.p_o), range(i[1]*hdmr.p_o, (i[1]+1)*hdmr.p_o)):
                b_m[:, col] = np.multiply(b_m[:, j[0]], b_m[:, j[1]])
                col += 1
    
    # Third order columns of basis matrix
    if max_order == 3:
        for i in _prod(range(0, hdmr.d-2), range(1, hdmr.d-1), range(2, hdmr.d)):
            if extended_base:
                b_m[:, col:col+hdmr.p_o] = b_m[:, i[0]*hdmr.p_o:(i[0]+1)*hdmr.p_o]
                col += hdmr.p_o
                b_m[:, col:col+hdmr.p_o] = b_m[:, i[1]*hdmr.p_o:(i[1]+1)*hdmr.p_o]
                col += hdmr.p_o
                b_m[:, col:col+hdmr.p_o] = b_m[:, i[2]*hdmr.p_o:(i[2]+1)*hdmr.p_o]
                col += hdmr.p_o
                b_m[:, col:col+hdmr.p_o**2] = b_m[:, hdmr.tnt1+(2*hdmr.nt1)*(i[0]+1)+i[0]*(hdmr.p_o**2):hdmr.tnt1+(i[0]+1)*(hdmr.p_o**2+2*hdmr.nt1)]
                col += hdmr.p_o**2
                b_m[:, col:col+hdmr.p_o**2] = b_m[:, hdmr.tnt1+(2*hdmr.nt1)*(i[1]+1)+i[1]*(hdmr.p_o**2):hdmr.tnt1+(i[1]+1)*(hdmr.p_o**2+2*hdmr.nt1)]
                col += hdmr.p_o**2
                b_m[:, col:col+hdmr.p_o**2] = b_m[:, hdmr.tnt1+(2*hdmr.nt1)*(i[2]+1)+i[2]*(hdmr.p_o**2):hdmr.tnt1+(i[2]+1)*(hdmr.p_o**2+2*hdmr.nt1)]
                col += hdmr.p_o**2

            for j in _prod(range(i[0]*hdmr.p_o, (i[0]+1)*hdmr.p_o), range(i[1]*hdmr.p_o, (i[1]+1)*hdmr.p_o), range(i[2]*hdmr.p_o, (i[2]+1)*hdmr.p_o)):
                b_m[:, col] = np.multiply(np.multiply(b_m[:, j[0]], b_m[:, j[1]]), b_m[:, j[2]])
                col += 1

    return b_m


def _orth_poly_coeff(X, hdmr):
    k = 0
    M = np.zeros((hdmr.p_o+1, hdmr.p_o+1, hdmr.d))
    for i in range(hdmr.d):
        k = 0
        for j in range(hdmr.p_o+1):
            for z in range(hdmr.p_o+1):
                M[j, z, i] = sum(X[:, i]**k) / hdmr.N
                k += 1
            k = j + 1
    coeff = np.zeros((hdmr.p_o, hdmr.p_o+1, hdmr.d))
    for i in range(hdmr.d):
        for j in range(hdmr.p_o):
            for k in range(j+2):
                z = list(range(j+2))
                z.pop(k)
                det_ij = det(M[:j+1,:j+1, i]) * det(M[:j+2, :j+2, i])
                coeff[j, j+1-k, i] = (-1)**(j+k+1) * det(M[:j+1, z, i]) / np.sqrt(det_ij)

    return coeff


def _prod(*inputs):
    seen = set()
    for prod in product(*inputs):
        prod_set = frozenset(prod)
        if len(prod_set) != len(prod):
            continue
        if prod_set not in seen:
            seen.add(prod_set)
            yield prod


def _fanova(b_m, hdmr, Y, bootstrap, max_order, subset, extended_base, max_iter, lambdax):
    for t in range(bootstrap):
        # Extract model output for a corresponding bootstrap iteration
        Y_idx = Y[hdmr.idx[:, t], 0]
        # Subtract mean from it
        Y_idx -= np.mean(Y_idx)
        
        if extended_base:
            cost = _cost_matrix(b_m[hdmr.idx[:, t], :], hdmr, subset, max_order)
            _d_morph(b_m[hdmr.idx[:, t], :], cost, Y_idx, bootstrap, hdmr)
        else:
            _first_order(b_m[hdmr.idx[:, t], :hdmr.tnt1], Y_idx, subset, max_iter, lambdax, hdmr)
            _second_order(b_m[hdmr.idx[:, t], hdmr.tnt1:hdmr.tnt1+hdmr.tnt2], Y_idx, subset, lambdax, hdmr)
            _third_order()

        # Calculate component functions         
        Y_e = _comp_func(b_m[hdmr.idx[:, t], :], solution, subset, hdmr, max_order)
        # Sensitivity Analysis
        _ancova(Y_idx, Y_e, hdmr, t)


def _cost_matrix(b_m, hdmr, bootstrap, max_order):
    cost = np.zeros((hdmr.a_tnt, hdmr.a_tnt))

    range_2nd_1 = lambda x: range(hdmr.tnt1+(x)*hdmr.nt2, hdmr.tnt1+(x+1)*hdmr.nt2)
    range_2nd_2 = lambda x: range(hdmr.tnt1+(x)*hdmr.nt2, hdmr.tnt1+(x)*hdmr.nt2+hdmr.p_o*2)
    range_3rd_1 = lambda x: range(hdmr.tnt1+hdmr.tnt2+(x)*hdmr.nt3, hdmr.tnt1+hdmr.tnt2+(x+1)*hdmr.nt3)
    range_3rd_2 = lambda x: range(hdmr.tnt1+hdmr.tnt2+(x)*hdmr.nt3, hdmr.tnt1+hdmr.tnt2+(x)*hdmr.nt3+3*hdmr.p_o+3*hdmr.p_o**2)
                                  
    if max_order > 1:
        sr_i = np.mean(b_m, axis=0, keepdims=True)
        sr_ij = np.zeros((2*hdmr.p_o+1, hdmr.nt2))
        ct = 0
        for _ in _prod(range(0, hdmr.d-1), range(1, hdmr.d)):
            sr_ij[0, :] = sr_i[0, range_2nd_1(ct)]
            sr_ij[1:, :] = (b_m[:, range_2nd_2(ct)].T @ b_m[:, range_2nd_1(ct)]) / bootstrap
            cost[np.ix_(range_2nd_1(ct), range_2nd_1(ct))] = sr_ij.T @ sr_ij
            ct += 1
    if max_order == 3:
        sr_ijk = np.zeros((3*hdmr.p_o+3*hdmr.p_o**2+1, hdmr.nt3))
        ct = 0
        for _ in _prod(range(0, hdmr.d-2), range(1, hdmr.d-1), range(2, hdmr.d)):
            sr_ijk[0, :] = sr_i[0, range_3rd_1(ct)]
            sr_ijk[1:, :] = b_m[:, range_3rd_2(ct)].T @ b_m[:, range_3rd_1(ct)]
            cost[np.ix_(range_3rd_1(ct), range_3rd_1(ct))] = sr_ijk.T @ sr_ijk
            ct += 1
    
    return cost


def _d_morph(b_m, cost, Y_idx, subset, hdmr):
    # Left Matrix Multiplication with the transpose of cost matrix
    a = (b_m.T @ b_m) / subset # LHS
    b = (b_m.T @ Y_idx) / subset # RHS
    try:
        # Pseudo-Inverse of LHS
        a_inv, rank = pinv(a, atol=max(a.shape)*np.finfo(np.float64).eps, return_rank=True)
        # Least-Square Solution
        x = a_inv @ b
        # Projection Matrix
        pr = np.eye(hdmr.a_tnt) - (a_inv @ a)
        pb = pr @ cost
        U, _, Vh = svd(pb)
    except LinAlgError:
        print("Pseudo-Inverse did not converge")
    
    nullity = b_m.shape[1] - rank
    V = Vh.T
    U = np.delete(U, range(0, nullity), axis=1)
    V = np.delete(V, range(0, nullity), axis=1)

    # D-Morph Regression Solution
    hdmr.x = V @ pinv(U.T @ V) @ U.T @ x


def _first_order(b_m1, Y_idx, subset, max_iter, lambdax, hdmr):
    """Compute first order component functions sequentially"""
    # Temporary first order component matrix
    Y_i = np.empty((subset, hdmr.nc1))
    # Initialize iter
    iter = 0
    # To increase readibility
    n1 = hdmr.nt1
    # L2 Penalty
    lambda_eye = lambdax * np.identity(n1)
    for i in range(hdmr.d):
        try:
            # Left hand side
            a = (b_m1[:, i*n1:n1*(i+1)].T @ b_m1[:, i*n1:n1*(i+1)]) / subset
            # Adding L2 Penalty (Ridge Regression)
            a += lambda_eye
            # Right hand side
            b = (b_m1[:, i*n1:n1*(i+1)].T @ Y_idx) / subset
            # Solution
            hdmr.x[i*n1:n1*(i+1)] = solve(a, b)
            # Component functions
            Y_i[:, i] = b_m1[:, i*n1:n1*(i+1)] @ hdmr.x[i*n1:n1*(i+1)]
        except LinAlgError:
            print("Least-square regression did not converge. Please increase L2 penalty term!")

    # Backfitting method
    var_old = np.square(hdmr.x[:hdmr.tnt1])
    while True:
        for i in range(hdmr.d):
            z = list(range(hdmr.d))
            z.remove(i)
            Y_res = Y_idx - np.sum(Y_i[:, z], axis=1)
            # Left hand side
            a = (b_m1[:, i*n1:n1*(i+1)].T @ b_m1[:, i*n1:n1*(i+1)]) / subset
            # Right hand side
            b = (b_m1[:, i*n1:n1*(i+1)].T @ Y_res) / subset
            # Solution
            hdmr.x[i*n1:n1*(i+1)] = solve(a, b)
            # Component functions
            Y_i[:, i] = b_m1[:, i*n1:n1*(i+1)] @ hdmr.x[i*n1:n1*(i+1)]

        var_max = np.absolute(var_old - np.square(hdmr.x[:hdmr.tnt1])).max()
        var_old = np.square(hdmr.x[:hdmr.tnt1])
        iter += 1

        if (var_max < 1e-4) or (iter > max_iter): 
            break


def _comp_func(b_m, x, subset, hdmr, max_order):
    Y_t = np.zeros((subset, hdmr.a_tnt))
    Y_e = np.zeros((subset, hdmr.nc_t))

    # Temporary matrix
    Y_t = np.multiply(b_m, np.tile(x, [subset, 1]))

    # First order component functions
    for i in range(hdmr.nc1):
        Y_e[:, i] = np.sum(Y_t[:, i*hdmr.p_o:(i+1)*hdmr.p_o], axis=1)

    # Second order component functions
    if max_order > 1:
        for i in range(hdmr.nc2):
            Y_e[:, hdmr.nc1+i] = np.sum(Y_t[:, hdmr.tnt1+(i)*hdmr.nt2:hdmr.tnt1+(i+1)*hdmr.nt2], axis=1)

    # Third order component functions
    if max_order == 3:
        for i in range(hdmr.nc3):
            Y_e[:, hdmr.nc1+hdmr.nc2+i] = np.sum(Y_t[:, hdmr.tnt1+hdmr.tnt2+(i)*hdmr.nt3:hdmr.tnt1+hdmr.tnt2+(i+1)*hdmr.nt3], axis=1)
        
    return Y_e


def _ancova(Y, Y_e, hdmr, t):
    """Analysis of Covariance."""
    # Compute the sum of all Y_em terms
    Y_sum = np.sum(Y_e, axis=1)

    # Total Variance
    tot_v = np.var(Y)

    # Analysis of covariance
    for j in range(hdmr.nc_t):
        # Covariance matrix of jth term of Y_em and actual Y
        c = np.cov(np.stack((Y_e[:, j], Y), axis=0))

        # Total sensitivity of jth term         ( = Eq. 19 of Li et al )
        hdmr.S[j, t] = c[0, 1] / tot_v

        # Covariance matrix of jth term with emulator Y without jth term
        c = np.cov(np.stack((Y_e[:, j], Y_sum - Y_e[:, j]), axis=0))

        # Structural contribution of jth term   ( = Eq. 20 of Li et al )
        hdmr.Sa[j, t] = c[0, 0] / tot_v

        # Correlative contribution of jth term  ( = Eq. 21 of Li et al )
        hdmr.Sb[j, t] = c[0, 1] / tot_v
