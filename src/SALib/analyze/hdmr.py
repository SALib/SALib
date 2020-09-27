from typing import Dict, List, Optional
from types import MethodType

import itertools
import time
import numpy as np
import pandas as pd
from SALib.plotting.hdmr import plot as hdmr_plot
from scipy import (stats, special, interpolate)

from . import common_args
from ..util import read_param_file, ResultDict
from matplotlib import pyplot as plt

import warnings


__all__ = ['analyze', 'cli_parse', 'cli_action']


def analyze(problem: Dict, X: np.array, Y: np.array, 
            maxorder: int = 2, maxiter: int = 100, 
            m: int = 2, K: int = 20, R: int = None, alpha: float = 0.95,
            lambdax: float = 0.01,
            print_to_console: bool = True, seed: int = None) -> Dict:
    """High-Dimensional Model Representation (HDMR) using B-spline functions.

    HDMR is used for variance-based global sensitivity analysis (GSA) with 
    correlated and uncorrelated inputs. This function uses as input

    - a N x d matrix of N different d-vectors of model inputs (factors/parameters)
    - a N x 1 vector of corresponding model outputs

    Returns to the user:
    - each factor's first, second, and third order sensitivity coefficient
      (separated in total, structural and correlative contributions), 
    - an estimate of their 95% confidence intervals (from bootstrap method)
    - the coefficients of the significant B-spline basis functions that
      govern output, 
    - Y (determined by an F-test of the error residuals of the HDMR model 
      (emulator) with/without a given first, second and/or
      third order B-spline). These coefficients define an emulator that can
      be used to predict the output, Y, of the original (CPU-intensive)
      model for any d-vector of model inputs. For uncorrelated model inputs
      (columns of X are independent), the HDMR sensitivity indices reduce
      to a single index (= structural contribution), consistent with their
      values derived from commonly used variance-based GSA methods.

    Parameters
    ----------
    problem : dict
        The problem definition

    X : numpy.matrix
        The NumPy matrix containing the model inputs, N rows by d columns

    Y : numpy.array
        The NumPy array containing the model outputs for each row of X

    maxorder : int (1-3, default: 2) 
        Maximum HDMR expansion order

    maxiter : int (1-1000, default: 100)
        Max iterations backfitting

    m : int (2-10, default: 2)
        Number of B-spline intervals

    K : int (1-100, default: 20)
        Number of bootstrap iterations

    R : int (100-N/2, default: N/2)
        Number of bootstrap samples. Will be set to length of `Y` if `K` is set to 1.

    alpha : float (0.5-1) 
        Confidence interval F-test

    lambdax : float (0-10, default: 0.01)
        Regularization term

    print_to_console : bool
        Print results directly to console (default False)
    
    seed : bool
        Set a seed value

    Returns
    -------
    Si : ResultDict,
        Sa: Uncorrelated contribution
        Sa_conf: Confidence interval of Sa
        Sb: Correlated contribution
        Sb_conf: Confidence interval of Sb
        S: Total contribution of a particular term
        S_conf: Confidence interval of S
        ST: Total contribution of a particular dimension/parameter
        ST_conf: Confidence interval of ST
        Sa: Uncorrelated contribution
        select: Number of selection (F-Test)
        Em: Result set
            C1: First order coefficient 
            C2: Second order coefficient
            C3: Third Order coefficient

    References
    ----------
    .. [1] Genyuan Li, H. Rabitz, P.E. Yelvington, O.O. Oluwole, F. Bacon,
            C.E. Kolb, and J. Schoendorf, "Global Sensitivity Analysis for
            Systems with Independent and/or Correlated Inputs", Journal of
            Physical Chemistry A, Vol. 114 (19), pp. 6022 - 6032, 2010,
            https://doi.org/10.1021/jp9096919
       
    Examples
    --------
        >>> X = saltelli.sample(problem, 1000)
        >>> Y = Ishigami.evaluate(X)
        >>> Si = hdmr.analyze(problem, X, Y, **options)

    Contributed by
    --------------
        @sahin-abdullah (sahina@uci.edu)
    """
    # Random Seed
    if seed:
        np.random.seed(seed)

    if problem.get('groups'):
        warnings.warn("SALib HDMR method does not currently support groups")

    # Initial part: Check input arguments and define HDMR variables
    settings = _check_settings(X, Y, maxorder, maxiter, m, K, R, alpha, lambdax)
    init_vars = _init(X, Y, settings)

    # Sensitivity Analysis Computation with/without bootstraping
    SA, Em, RT, Y_em, idx = _compute(X, Y, settings, init_vars)

    # Finalize results
    Si = _finalize(problem, SA, Em, problem['num_vars'], alpha, maxorder, RT, Y_em, idx, X, Y)

    # Print results to console
    if print_to_console:
        _print(Si, problem['num_vars'])

    return Si

def _check_settings(X, Y, maxorder, maxiter, m, K, R, alpha, lambdax):
    """Perform checks to ensure all parameters are within usable/expected ranges."""
    # Get dimensionality of numpy arrays
    N, d = X.shape
    y_row = Y.shape[0]

    # Now check input-output mismatch
    if d == 1:
        raise RuntimeError("Matrix X contains only a single column: No point to do sensitivity analysis when d = 1.")
    if N < 300:
        raise RuntimeError(f"Number of samples in matrix X, {N}, is insufficient. Need at least 300.")
    if N != y_row:
        raise RuntimeError(f"Dimension mismatch. The number of outputs ({y_row}) should match number of samples ({N})")
    if Y.size != N:
        raise RuntimeError("Y should be a N x 1 vector with one simulated output for each N parameter vectors.")

    if maxorder not in (1,2,3):
        raise RuntimeError(f"Field \"maxorder\" of options should be an integer with values of 1, 2 or 3, got {maxorder}")

    # Important next check for maxorder - as maxorder relates to d
    if (d == 2) and (maxorder > 2):
        raise RuntimeError("SALib-HDMR ERRROR: Field \"maxorder\" of options has to be 2 as d = 2 (X has two columns)")

    if maxiter not in np.arange(1, 1001):
        raise RuntimeError("Field \"maxiter\" of options should be an integer between 1 to 1000.")
    
    if m not in np.arange(1, 11):
        raise RuntimeError("Field \"m\" of options should be an integer between 1 to 10.")

    if K not in np.arange(1, 101):
        raise RuntimeError("Field \"K\" of options should be an integer between 1 to 100.")

    if R is None:
        R = y_row // 2
    elif R not in np.arange(300, N+1):
        raise RuntimeError(f"Field \"R\" of options should be an integer between 300 and {N}, the number of rows matrix X.")

    if (K == 1) and (R != y_row):
        R = y_row
    
    if alpha < 0.5 or alpha > 1.0:
        raise RuntimeError("Field \"alpha\" of options should be an integer between 0.5 to 1.0")

    if lambdax > 10.0:
        raise RuntimeError("SALib-HDMR WARNING: Field \"lambdax\" of options set rather large. Default: lambdax = 0.01")
    elif lambdax < 0.0:
        raise RuntimeError("Field \"lambdax\" (regularization term) of options cannot be smaller than zero. Default: 0.01")

    return [N, d, maxorder, maxiter,
            m, K, R, alpha, lambdax]


def _compute(X, Y, settings, init_vars):
    """Compute HDMR"""
    N, d, maxorder, maxiter, m, K, R, alpha, lambdax = settings
    Em, idx, SA, RT, Y_em, Y_id, m1, m2, m3, j1, j2, j3 = init_vars

    # DYNAMIC PART: Bootstrap for confidence intervals
    for k in range(K):
        tic = time.time()  # Start timer

        # Extract the "right" Y values
        Y_id[:, 0] = Y[idx[:, k]]

        # Compute the variance of the model output
        SA['V_Y'][k, 0] = np.var(Y_id)

        # Mean of the output
        Em['f0'][k] = np.sum(Y_id[:, 0]) / R

        # Compute residuals
        Y_res = Y_id - Em['f0'][k]

        # 1st order component functions: ind/backfitting
        Y_em[:, j1], Y_res, Em['C1'][:, :, k] = _first_order(Em['B1'][idx[:, k], :, :], Y_res, 
                                                             Em['C1'][:, :, k], R, Em['n1'], m1,
                                                             maxiter, lambdax)

        # 2nd order component functions: individual
        if (maxorder > 1):
            Y_em[:, j2], Y_res, Em['C2'][:, :, k] = _second_order(Em['B2'][idx[:, k], :, :], Y_res, 
                                                                  Em['C2'][:, :, k], R, Em['n2'],
                                                                  m2, lambdax)

        # 3rd order component functions: individual
        if (maxorder == 3):
            Y_em[:, j3], Em['C3'][:, :, k] = _third_order(Em['B3'][idx[:, k], :, :], Y_res, 
                                                          Em['C3'][:, :, k], R, Em['n3'], m3, lambdax)

        # Identify significant and insignificant terms
        Em['select'][:, k] = f_test(Y_id, Em['f0'][k], Y_em, R, alpha, m1, m2, m3, Em['n1'], Em['n2'],
                                    Em['n3'], Em['n'])

        # Store the emulated output
        Em['Y_e'][:, k] = Em['f0'][k] + np.sum(Y_em, axis=1)

        # RMSE of emulator
        Em['RMSE'][k] = np.sqrt(np.sum(np.square(Y_id - Em['Y_e'][:, k])) / R)

        # Compute sensitivity indices
        SA['S'][:, k], SA['Sa'][:, k], SA['Sb'][:, k] = ancova(Y_id, Y_em, SA['V_Y'][k],
                                                               R, Em['n'])

        RT[0, k] = time.time() - tic  # Compute CPU time kth emulator

    return (SA, Em, RT, Y_em, idx)


def _init(X, Y, settings):
    """Initialize HDMR variables."""
    N, d, maxorder, maxiter, m, K, R, alpha, lambdax = settings

    # Setup Bootstrap (if K > 1)
    if (K == 1):
        # No Bootstrap
        idx = np.arange(0, N).reshape(N, 1)
    else:
        # Now setup the boostrap matrix with selection matrix, idx, for samples
        idx = np.argsort(np.random.rand(N, K), axis=0)[:R]

    # Compute normalized X-values
    X_n = (X - np.tile(X.min(0), (N, 1))) / \
        np.tile((X.max(0)) - X.min(0), (N, 1))

    # DICT Em: Important variables
    n2 = 0
    n3 = 0
    c2 = np.empty
    c3 = np.empty

    # determine all combinations of parameters (coefficients) for each order
    c1 = np.arange(0, d)
    n1 = d

    # now return based on maxorder used
    if maxorder > 1:
        c2 = np.asarray(list(itertools.combinations(np.arange(0, d), 2)))
        n2 = c2.shape[0]
    if maxorder == 3:
        c3 = np.asarray(list(itertools.combinations(np.arange(0, d), 3)))
        n3 = c3.shape[0]

    # calulate total number of coefficients
    n = n1 + n2 + n3

    # Initialize m1, m2 and m3 - number of coefficients first, second, third
    # order
    m1 = m + 3
    m2 = m1**2
    m3 = m1**3

    # STRUCTURE Em: Initialization
    Em = {'nterms': np.zeros(K), 'RMSE': np.zeros(K), 'm': m, 'Y_e': np.zeros((R, K)), 'f0': np.zeros(K),
            'c1': c1, 'C1': np.zeros((m1, n1, K)), 'C2': np.zeros((1, 1, K)), 'C3': np.zeros((1, 1, K)), 
            'n': n, 'n1': n1, 'n2': n2, 'n3': n3, 'maxorder': maxorder, 'select': np.zeros((n, K)),
            'B1': np.zeros((N, m1, n1))}

    if (maxorder >= 2):
        Em.update({'c2': c2, 'B2': np.zeros((N, m2, n2)), 'C2': np.zeros((m2, n2, K))})
    if (maxorder >= 3):
        Em.update({'c3': c3, 'B3': np.zeros((N, m3, n3)), 'C3': np.zeros((m3, n3, K))})

    # Compute cubic B-spline
    Em['B1'] = B_spline(X_n, m, d)

    # Now compute B values for second order
    if (maxorder > 1):
        beta = np.array(list(itertools.product(range(m1), repeat=2)))
        for k, j in itertools.product(range(n2), range(m2)):
            Em['B2'][:, j, k] = np.multiply(
                Em['B1'][:, beta[j, 0], Em['c2'][k, 0]], 
                Em['B1'][:, beta[j, 1], Em['c2'][k, 1]])

    # Compute B values for third order
    if (maxorder == 3):
        # Now compute B values for third order
        beta = np.array(list(itertools.product(range(m1), repeat=3)))
        EmB1 = Em['B1']

        for k, j in itertools.product(range(n3), range(m3)):
            Emc3_k = Em['c3'][k]
            beta_j = beta[j]
            Em['B3'][:, j, k] = np.multiply(np.multiply(EmB1[:, beta_j[0], Emc3_k[0]], 
                                                        EmB1[:, beta_j[1], Emc3_k[1]]),
                                            EmB1[:, beta_j[2], Emc3_k[2]])

    # DICT SA: Sensitivity analysis and analysis of variance decomposition
    em_k = np.zeros((Em['n'], K))
    SA = {
        'S': em_k,
        'Sa': em_k.copy(),
        'Sb': em_k.copy(),
        'ST': np.zeros((d, K)),
        'V_Y': np.zeros((K, 1))
    }

    # Return runtime
    RT = np.zeros((1, K))

    # Initialize emulator matrix
    Y_em = np.zeros((R, Em['n']))

    # Initialize index of first, second and third order terms (columns of Y_em)
    j1 = range(n1)
    j2 = range(n1, n1 + n2)
    j3 = range(n1 + n2, n1 + n2 + n3)

    # Initialize temporary Y_id for bootstrap
    Y_id = np.zeros((R, 1))

    return [Em, idx, SA, RT, Y_em, Y_id, m1, m2, m3, j1, j2, j3]


def B_spline(X, m, d):
    """Generate cubic B-splines using scipy basis_element method. 
    
    Knot points (`t`) are automatically determined.

    References
    ----------
    .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.basis_element.html
    """
    # Initialize B matrix
    B = np.zeros((X.shape[0], m+3, d))

    # Cubic basis-spline settings
    k = np.arange(-1, m+2)

    # Compute B-Splines
    for j, i in itertools.product(range(d), range(m+3)):
        k_i = k[i]
        t = np.arange(k_i-2, k_i+3) / m
        temp = interpolate.BSpline.basis_element(t)(X[:,j]) * np.power(m,3)
        B[:, i, j] = np.where(temp < 0, 0, temp)
    
    return B


def _first_order(B1, Y_res, C1, R, n1, m1, maxiter, lambdax):
    """Compute first order sensitivities."""
    Y_i = np.zeros((R, n1))       # Initialize 1st order contributions
    T1 = np.zeros((m1, R, n1))    # Initialize T(emporary) matrix - 1st
    it = 0                        # Initialize iteration counter

    # First order individual estimation
    for j in range(n1):
        # Regularized least squares inversion ( minimize || C1 ||_2 )
        B11 = np.matmul(np.transpose(B1[:, :, j]), B1[:, :, j])

        if (np.linalg.det(B11) == 0):
            C1[:, j] = 0  # Ill-defined
        else:
            T1[:, :, j] = np.linalg.solve(np.add(B11, np.multiply(
                lambdax, np.identity(m1))), np.transpose(B1[:, :, j]))

        C1[:, j] = np.matmul(T1[:, :, j], Y_res).reshape(m1,)
        Y_i[:, j] = np.matmul(B1[:, :, j], C1[:, j])

    # Backfitting Method
    var1b_old = np.sum(np.square(C1), axis=0)
    varmax = 1
    while (varmax > 1e-3) and (it < maxiter):
        for j in range(n1):
            Y_r = Y_res
            for z in range(n1):
                if j != z:
                    Y_r = np.subtract(Y_r, np.matmul(
                        B1[:, :, z], C1[:, z]).reshape(R, 1))

            C1[:, j] = np.matmul(T1[:, :, j], Y_r).reshape(m1,)

        var1b_new = np.sum(np.square(C1), axis=0)
        varmax = np.max(np.absolute(np.subtract(var1b_new, var1b_old)))
        var1b_old = var1b_new
        it += 1

    # Now compute first-order terms
    for j in range(n1):
        Y_i[:, j] = np.matmul(B1[:, :, j], C1[:, j])

        # Subtract each first order term from residuals
        Y_res = np.subtract(Y_res, Y_i[:, j].reshape(R, 1))

    return (Y_i, Y_res, C1)


def _second_order(B2, Y_res, C2, R, n2, m2, lambdax):
    """Compute second order sensitivities."""
    Y_ij = np.zeros((R, n2))      # Initialize 1st order contributions
    T2 = np.zeros((m2, R, n2))    # Initialize T(emporary) matrix - 1st

    # First order individual estimation
    for j in range(n2):
        # Regularized least squares inversion ( minimize || C1 ||_2 )
        B22 = np.matmul(np.transpose(B2[:, :, j]), B2[:, :, j])

        if (np.linalg.det(B22) == 0):
            C2[:, j] = 0  # Ill-defined
        else:
            T2[:, :, j] = np.linalg.solve(np.add(B22, np.multiply(
                lambdax, np.identity(m2))), np.transpose(B2[:, :, j]))

        C2[:, j] = np.matmul(T2[:, :, j], Y_res).reshape(m2,)
        Y_ij[:, j] = np.matmul(B2[:, :, j], C2[:, j])

    # Now compute second-order terms
    for j in range(n2):
        Y_ij[:, j] = np.matmul(B2[:, :, j], C2[:, j])

        # Subtract each first order term from residuals
        Y_res = np.subtract(Y_res, Y_ij[:, j].reshape(R, 1))

    return (Y_ij, Y_res, C2)


def _third_order(B3, Y_res, C3, R, n3, m3, lambdax):
    """Compute third order sensitivities."""
    Y_ijk = np.zeros((R, n3))      # Initialize 1st order contributions
    T3 = np.zeros((m3, R, n3))     # Initialize T(emporary) matrix - 1st

    # First order individual estimation
    for j in range(n3):
        # Regularized least squares inversion ( minimize || C1 ||_2 )
        B33 = np.matmul(np.transpose(B3[:, :, j]), B3[:, :, j])
        if (np.linalg.det(B33) == 0):
            C3[:, j] = 0  # Ill-defined
        else:
            T3[:, :, j] = np.linalg.solve(np.add(B33, np.multiply(
                lambdax, np.identity(m3))), np.transpose(B3[:, :, j]))

        C3[:, j] = np.matmul(T3[:, :, j], Y_res).reshape(m3,)
        Y_ijk[:, j] = np.matmul(B3[:, :, j], C3[:, j])

    return (Y_ijk, C3)


def f_test(Y, f0, Y_em, R, alpha, m1, m2, m3, n1, n2, n3, n):
    """Use F-test for model selection."""
    # Initialize ind with zeros (all terms insignificant)
    select = np.zeros((n, 1))

    # Determine the significant components of the HDMR model via the F-test
    Y_res0 = Y - f0
    SSR0 = np.sum(np.square(Y_res0))
    p0 = 0
    for i in range(n):
        # model with ith term included
        Y_res1 = Y_res0 - Y_em[:, i].reshape(R, 1)

        # Number of parameters of proposed model (order dependent)
        if i <= n1:
            p1 = m1        # 1st order
        elif i > n1 and i <= (n1 + n2):
            p1 = m2        # 2nd order
        else:
            p1 = m3        # 3rd order

        # Calculate SSR of Y1
        SSR1 = np.sum(np.square(Y_res1))

        # Now calculate the F_stat (F_stat > 0 -> SSR1 < SSR0 )
        F_stat = ((SSR0 - SSR1) / (p1 - p0)) / (SSR1 / (R - p1))

        # Now calculate critical F value
        F_crit = stats.f.ppf(q=alpha, dfn=p1 - p0, dfd=R - p1)

        # Now determine whether to accept ith component into model
        if F_stat > F_crit:
            # ith term is significant and should be included in model
            select[i] = 1

    return select.reshape(n,)


def ancova(Y, Y_em, V_Y, R, n):
    """Analysis of Covariance."""
    # Compute the sum of all Y_em terms
    Y0 = np.sum(Y_em[:, :], axis=1)

    # Initialize each variable
    S, S_a, S_b = [np.zeros((n,)) for _ in range(3)]

    # Analysis of covariance
    for j in range(n):
        # Covariance matrix of jth term of Y_em and actual Y
        C = np.cov(np.stack((Y_em[:, j], Y.reshape(R,)), axis=0))

        # Total sensitivity of jth term         ( = Eq. 19 of Li et al )
        S[j] = C[0, 1] / V_Y

        # Covariance matrix of jth term with emulator Y without jth term
        C = np.cov(np.stack((Y_em[:, j], Y0 - Y_em[:, j]), axis=0))

        # Structural contribution of jth term   ( = Eq. 20 of Li et al )
        S_a[j] = C[0, 0] / V_Y
        
        # Correlative contribution of jth term  ( = Eq. 21 of Li et al )
        S_b[j] = C[0, 1] / V_Y

    return (S, S_a, S_b)


def _finalize(problem, SA, Em, d, alpha, maxorder, RT, Y_em, bootstrap_idx, X, Y):
    """Creates ResultDict to be returned."""

    # Create Sensitivity Indices Result Dictionary
    keys = ('Sa', 'Sa_conf', 'Sb', 'Sb_conf', 'S', 'S_conf', 'select',
            'Sa_sum', 'Sa_sum_conf', 'Sb_sum', 'Sb_sum_conf', 'S_sum', 'S_sum_conf')
    Si = ResultDict((k, np.zeros(Em['n'])) for k in keys)
    Si['Term'] = [None] * Em['n']
    Si['ST'] = [np.nan, ] * Em['n']
    Si['ST_conf'] = [np.nan, ] * Em['n']

    # Z score
    def z(p): return (-1) * np.sqrt(2) * special.erfcinv(p * 2)

    # Multiplier for confidence interval
    mult = z(alpha + (1 - alpha) / 2)

    # Compute the total sensitivity of each parameter/coefficient
    for r in range(Em['n1']):
        if maxorder == 1:
            TS = SA['S'][r, :]
        elif maxorder == 2:
            ij = Em['n1'] + np.where(np.sum(Em['c2'] == r, axis=1) == 1)[0]
            TS = np.sum(SA['S'][np.append(r, ij), :], axis=0)
        elif maxorder == 3:
            ij = Em['n1'] + np.where(np.sum(Em['c2'] == r, axis=1) == 1)[0]
            ijk = Em['n1'] + Em['n2'] + \
                np.where(np.sum(Em['c3'] == r, axis=1) == 1)[0]
            TS = np.sum(SA['S'][np.append(r, np.append(ij, ijk)), :], axis=0)
        Si['ST'][r] = np.mean(TS)
        Si['ST_conf'][r] = mult * np.std(TS)

    # Fill Expansion Terms
    ct = 0
    for i in range(Em['n1']):
        Si['Term'][ct] = problem['names'][i]
        ct += 1

    for i in range(Em['n2']):
        Si['Term'][ct] = '/'.join([problem['names'][Em['c2'][i, 0]],
                                   problem['names'][Em['c2'][i, 1]]])
        ct += 1

    for i in range(Em['n3']):
        Si['Term'][ct] = '/'.join([problem['names'][Em['c3'][i, 0]],
                                   problem['names'][Em['c3'][i, 1]], problem['names'][Em['c3'][i, 2]]])
        ct += 1

    # Assign Bootstrap Results to Si Dict
    Si['Sa'] = np.mean(SA['Sa'], axis=1)
    Si['Sb'] = np.mean(SA['Sb'], axis=1)
    Si['S'] = np.mean(SA['S'], axis=1)
    Si['Sa_sum'] = np.mean(np.sum(SA['Sa'], axis=0))
    Si['Sb_sum'] = np.mean(np.sum(SA['Sb'], axis=0))
    Si['S_sum'] = np.mean(np.sum(SA['S'], axis=0))

    # Compute Confidence Interval
    Si['Sa_conf'] = mult * np.std(SA['Sa'], axis=1)
    Si['Sb_conf'] = mult * np.std(SA['Sb'], axis=1)
    Si['S_conf'] = mult * np.std(SA['S'], axis=1)
    Si['Sa_sum_conf'] = mult * np.std(np.sum(SA['Sa']))
    Si['Sb_sum_conf'] = mult * np.std(np.sum(SA['Sb']))
    Si['S_sum_conf'] = mult * np.std(np.sum(SA['S']))

    # F-test # of selection to print out
    Si['select'] = Em['select'].flatten()

    Si['names'] = Si['Term']

    # Additional data for plotting.
    Si['Em'] = Em
    Si['RT'] = RT
    Si['Y_em'] = Y_em
    Si['idx'] = bootstrap_idx
    Si['X'] = X
    Si['Y'] = Y

    Si.problem = problem
    Si.to_df = MethodType(to_df, Si)

    Si._plot = Si.plot
    Si.plot = MethodType(plot, Si)

    Si.emulate = MethodType(emulate, Si)

    return Si


def emulate(self, X, Y=None):
    '''Emulates model output with new input data. 
    
    Generates B-Splines with new input matrix, X, and multiplies it with 
    coefficient matrix, C. 
    
    Compares emulated results with observed vector, Y.

    Returns
    ========
    Si : ResultDict, Updated

    '''
    # Dimensionality
    N, d = X.shape

    # Expand C
    Em = self['Em']
    C1, C2, C3 = Em['C1'], Em['C2'], Em['C3']

    # Take average coefficient
    C1 = np.mean(C1, axis=2)
    C2 = np.mean(C2, axis=2)
    C3 = np.mean(C3, axis=2)

    # Get maxorder and other information from C matrix
    m = C1.shape[0] - 3
    m1 = m+3
    n1, n2, n3 = d, 0, 0
    if C2.shape[0] != 1:
        maxorder = 2
        m2 = C2.shape[0]
        c2 = np.asarray(list(itertools.combinations(np.arange(0, d), 2)))
        n2 = c2.shape[0]
    if C3.shape[0] != 1:
        maxorder = 3
        m3 = C3.shape[0]
        c3 = np.asarray(list(itertools.combinations(np.arange(0, d), 3)))
        n3 = c3.shape[0]
    
    # Initialize emulated Y's
    Y_em = np.zeros((N, n1+n2+n3))

    # Compute normalized X-values
    X_n = (X - np.tile(X.min(0), (N, 1))) / \
        np.tile((X.max(0)) - X.min(0), (N, 1))

    # Now compute B values
    B1 = B_spline(X_n, m, d)
    if (maxorder > 1):
        B2 = np.zeros((N, m2, n2))
        beta = np.array(list(itertools.product(range(m1), repeat=2)))
        for k, j in itertools.product(range(n2), range(m2)):
            B2[:, j, k] = np.multiply(B1[:, beta[j, 0], c2[k, 0]], 
                                      B1[:, beta[j, 1], c2[k, 1]])
    if (maxorder == 3):
        B3 = np.zeros((N, m3, n3))
        beta = np.array(list(itertools.product(range(m1), repeat=3)))
        for k, j in itertools.product(range(n3), range(m3)):
            Emc3_k = c3[k]
            beta_j = beta[j]
            B3[:, j, k] = np.multiply(np.multiply(B1[:, beta_j[0], Emc3_k[0]], 
                                                        B1[:, beta_j[1], Emc3_k[1]]),
                                            B1[:, beta_j[2], Emc3_k[2]])

    # Calculate emulated output: First Order
    for j in range(0, n1):
        Y_em[:, j] = np.matmul(B1[:, :, j], C1[:, j])

    # Second Order
    if maxorder > 1:
        for j in range(n1, n1+n2):
            Y_em[:, j] = np.matmul(B2[:, :, j-n1], C2[:, j-n1])

    # Third Order
    if maxorder == 3:
        for j in range(n1+n2, n1+n2+n3):
            Y_em[:, j] = np.matmul(B3[:, :, j-n1-n2], C3[:, j-n1-n2])

    Y_mean = 0
    if Y is not None:
        Y_mean = np.mean(Y)
        self['Y_test'] = Y

    emulated = np.sum(Y_em, axis=1)+Y_mean
    self['emulated'] = emulated


def to_df(self):
    '''Conversion method to Pandas DataFrame. To be attached to ResultDict.

    Returns
    ========
    Pandas DataFrame
    '''
    names = self['names']

    # Only convert these elements in dict to DF
    include_list = ['Sa', 'Sb', 'S', 'ST']
    include_list += [f'{name}_conf' for name in include_list]
    new_spec = {k: v for k, v in self.items() if k in include_list}

    return pd.DataFrame(new_spec, index=names)


def plot(self):
    """HDMR specific plotting.

    Gets attached to the SALib problem spec.
    """
    return hdmr_plot(self)


def _print(Si, d):
    print("\n")
    print(
        'Term    \t      Sa            Sb             S             ST         #select ')
    print('------------------------------------------------------------------------------------')

    format1 = (
        "%-11s   \t %5.2f (\261%.2f) %5.2f (\261%.2f) %5.2f (\261%.2f) %5.2f (\261%.2f)    %-3.0f")
    format2 = (
        "%-11s   \t %5.2f (\261%.2f) %5.2f (\261%.2f) %5.2f (\261%.2f)                  %-3.0f")

    for i in range(len(Si['Sa'])):
        if i < d:
            print(format1 % (Si['Term'][i], Si['Sa'][i], Si['Sa_conf'][i], Si['Sb'][i], Si['Sb_conf'][i], Si['S'][i],
                                Si['S_conf'][i], Si['ST'][i], Si['ST_conf'][i], np.sum(Si['select'][i])))
        else:
            print(format2 % (Si['Term'][i], Si['Sa'][i], Si['Sa_conf'][i], Si['Sb'][i], Si['Sb_conf'][i], Si['S'][i],
                                Si['S_conf'][i], np.sum(Si['select'][i])))

    print('------------------------------------------------------------------------------------')

    format3 = (
        "%-11s   \t %5.2f (\261%.2f) %5.2f (\261%.2f) %5.2f (\261%.2f)")
    print(format3 % ('Sum', Si['Sa_sum'], Si['Sa_sum_conf'],
                        Si['Sb_sum'], Si['Sb_sum_conf'], Si['S_sum'], Si['S_sum_conf']))
    
    keys = ('Sa_sum', 'Sb_sum', 'S_sum', 'Sa_sum_conf', 'Sb_sum_conf', 'S_sum_conf')
    for k in keys:
        Si.pop(k, None)

    return Si


def cli_parse(parser):
    parser.add_argument('-X', '--model-input-file', type=str, required=True,
                        default=None,
                        help='Model input file')
    parser.add_argument('-mor', '--maxorder', type=int, required=False,
                        default=2,
                        help='Maximum order of expansion 1-3')
    parser.add_argument('-mit', '--maxiter', type=int, required=False,
                        default=100,
                        help='Maximum iteration number')
    parser.add_argument('-m', '--m-int', type=int, required=False,
                        default=2,
                        help='B-spline interval')
    parser.add_argument('-K', '--K-bootstrap', type=int, required=False,
                        default=20,
                        help='Number of bootstrap')
    parser.add_argument('-R', '--R-subsample', type=int, required=False,
                        default=None,
                        help='Subsample size')
    parser.add_argument('-a', '--alpha', type=float, required=False,
                        default=0.95,
                        help='Confidence interval')
    parser.add_argument('-lambda', '--lambdax', type=float, required=False,
                        default=0.01,
                        help='Regularization constant')
    return parser


def cli_action(args):
    problem = read_param_file(args.paramfile)
    Y = np.loadtxt(args.model_output_file,
                   delimiter=args.delimiter, usecols=(args.column,))
    X = np.loadtxt(args.model_input_file, delimiter=args.delimiter, ndmin=2)
    mor = args.maxorder
    mit = args.maxiter
    m = args.m_int
    K = args.K_bootstrap
    R = args.R_subsample
    alpha = args.alpha
    lambdax = args.lambdax
    options = options = {'maxorder': mor, 'maxiter': mit, 'm': m,
                         'K': K, 'R': R, 'alpha': alpha, 'lambdax': lambdax, 'print_to_console': True}
    
    if len(X.shape) == 1:
        X = X.reshape((len(X), 1))

    analyze(problem, X, Y, **options)


if __name__ == "__main__":
    common_args.run_cli(cli_parse, cli_action)
