import time
import random
import numpy as np
import openturns as ot

from ..util import read_param_file
from ..util.distrs import make_input_distribution


def condMVN_new(cov, dependent_ind, given_ind, X_given):
    """ Returns conditional mean and variance of X[dependent.ind] | X[given.ind] = X.given
    where X is multivariateNormal(mean = mean, covariance = cov)"""
    
    cov = np.asarray(cov)
    
    B = cov[:, dependent_ind]
    B = B[dependent_ind]
    
    C = cov[:, dependent_ind]
    C = C[given_ind]
    
    D = cov[:, given_ind]
    D = D[given_ind]
    
    CDinv = np.dot(np.transpose(C), np.linalg.inv(D))
    
    condMean = np.dot(CDinv, X_given)
    condVar = B - np.dot(CDinv, C)
    condVar = ot.CovarianceMatrix(condVar)
    
    return condMean, condVar


def condMVN(mean, cov, dependent_ind, given_ind, X_given):
    """ Returns conditional mean and variance of X[dependent.ind] | X[given.ind] = X.given
    where X is multivariateNormal(mean = mean, covariance = cov)"""
    
    cov = np.array(cov)
    
    B = cov[:, dependent_ind]
    B = B[dependent_ind]
    
    C = cov[:, dependent_ind]
    C = C[given_ind]
    
    D = cov[:, given_ind]
    D = D[given_ind]
    
    CDinv = np.dot(np.transpose(C), np.linalg.inv(D))
    
    condMean = mean[dependent_ind] + np.dot(CDinv, (X_given - mean[given_ind]))
    condVar = B - np.dot(CDinv, C)
    condVar = ot.CovarianceMatrix(condVar)
    
    return condMean, condVar


def cond_sampling_new(distribution, n_sample, idx, idx_c, x_cond):
    """
    """
    margins_dep = [distribution.getMarginal(int(i)) for i in idx]
    margins_cond = [distribution.getMarginal(int(i)) for i in idx_c]

    # Creates a conditioned variables that follows a Normal distribution
    u_cond = np.zeros(x_cond.shape)
    for i, marginal in enumerate(margins_cond):
        u_cond[i] = np.asarray(ot.Normal().computeQuantile(marginal.computeCDF(x_cond[i])))

    sigma = np.asarray(distribution.getCopula().getCorrelation())
    cond_mean, cond_var = condMVN_new(sigma, idx, idx_c, u_cond)
    
    n_dep = len(idx)
    dist_cond = ot.Normal(cond_mean, cond_var)
    sample_norm = np.asarray(dist_cond.getSample(int(n_sample)))
    sample_x = np.zeros((n_sample, n_dep))
    phi = lambda x: ot.Normal().computeCDF(x)
    for i in range(n_dep):
        u_i = np.asarray(phi(sample_norm[:, i].reshape(-1, 1))).ravel()
        sample_x[:, i] = np.asarray(margins_dep[i].computeQuantile(u_i)).ravel()

    return sample_x


def cond_sampling(distribution, n_sample, idx, idx_c, x_cond):
    """
    """
    cov = np.asarray(distribution.getCovariance())
    mean = np.asarray(distribution.getMean())
    cond_mean, cond_var = condMVN(mean, cov, idx, idx_c, x_cond)
    dist_cond = ot.Normal(cond_mean, cond_var)
    sample = dist_cond.getSample(n_sample)
    return sample


def sub_sampling(distribution, n_sample, idx):
    """Sampling from a subset of a given distribution.

    The function takes the margin and correlation matrix subset and creates a new copula
    and distribution function to sample.

    Parameters
    ----------


    Returns
    -------
    sample : array,
        The sample of the subset distribution.
    """
    # Margins of the subset
    margins_sub = [distribution.getMarginal(int(j)) for j in idx]
    # Get the correlation matrix
    sigma = np.asarray(distribution.getCopula().getCorrelation())
    # Takes only the subset of the correlation matrix
    copula_sub = ot.NormalCopula(ot.CorrelationMatrix(sigma[:, idx][idx, :]))
    # Creates the subset distribution
    dist_sub = ot.ComposedDistribution(margins_sub, copula_sub)
    # Sample
    sample = np.asarray(dist_sub.getSample(int(n_sample)))
    return sample


def sample(problem, n_perms, n_var, n_outer, n_inner, randomize=True):
    """
    Parameters:
    -----------
    n_perms : int or None
        The number of permutations. If None, the exact permutations method
        is considerd.
        
    n_var : int
        The sample size for the output variance estimation.
        
    n_outer : int
        The number of conditionnal variance estimations.

    n_inner : int
        The sample size for the conditionnal output variance estimation.


    Should be in problem object:
    ----------------------------
        bounds: array-like
            Used for uniform case

        corr: array-like or None
            Correlation matrix

        seed: int [42]
            Is needed for synchronization of sample and analyze for Shapley
            while permutations generation.
         
        randomize: bool [True]
            Make samples random after seed manipulation for sample-analyze
            synchronization.


    Returns:
    --------
        X: np.array - samples 

    """
    assert isinstance(problem, dict), "problem should be of dict type"
    dim = problem['num_vars']
    seed = problem.get('seed', 42)

    assert isinstance(n_perms, (int, type(None))), \
        "The number of permutation should be an integer or None."
    assert isinstance(n_var, int), "n_var should be an integer."
    assert isinstance(n_outer, int), "n_outer should be an integer."
    assert isinstance(n_inner, int), "n_inner should be an integer."
    if isinstance(n_perms, int):
        assert n_perms > 0, "The number of permutation should be positive"
        
    assert n_var > 0, "n_var should be positive"
    assert n_outer > 0, "n_outer should be positive"
    assert n_inner > 0, "n_inner should be positive"


    input_distribution = make_input_distribution(problem)
    
    if n_perms is None:
        ot.RandomGenerator_SetSeed(seed)
        np.random.seed(seed)
        random.seed(seed)
        perms = list(ot.KPermutations(dim, dim).generate())
        n_perms = len(perms)
    else:
        ot.RandomGenerator_SetSeed(seed)
        np.random.seed(seed)
        random.seed(seed)
        perms = [np.random.permutation(dim) for i in range(n_perms)]


    if randomize:
        ot.RandomGenerator_SetSeed(int(time.time()))
        np.random.seed(int(time.time()))
        random.seed(int(time.time()))

    # Creation of the design matrix
    input_sample_1 = np.asarray(input_distribution.getSample(n_var))
    input_sample_2 = np.zeros((n_perms * (dim - 1) * n_outer * n_inner, dim))

    for i_p, perm in enumerate(perms):
        idx_perm_sorted = np.argsort(perm)  # Sort the variable ids
        for j in range(dim - 1):
            # Normal set
            idx_j = perm[:j + 1]
            # Complementary set
            idx_j_c = perm[j + 1:]
            sample_j_c = sub_sampling(input_distribution, n_outer, idx_j_c)
            for l, xjc in enumerate(sample_j_c):
                # Sampling of the set conditionally to the complementary
                # element
                xj = cond_sampling_new(input_distribution, n_inner, idx_j, idx_j_c, xjc)
                xx = np.c_[xj, [xjc] * n_inner]
                ind_inner = i_p * (dim - 1) * n_outer * n_inner + j * n_outer * n_inner + l * n_inner
                input_sample_2[ind_inner:ind_inner + n_inner, :] = xx[:, idx_perm_sorted]

    X = np.r_[input_sample_1, input_sample_2]
    return X



def cli_parse(parser):
    """Add method specific options to CLI parser.

    Parameters
    ----------
    parser : argparse object

    Returns
    ----------
    Updated argparse object
    """
    parser.add_argument('--n-perms', type=int, required=False, default=None,
                        help='Number of permutations. Should be less equal than factorial(dimension).')
    parser.add_argument('--n-outer', type=int, required=True,
                        help='Number of outer conditional samples.')
    parser.add_argument('--n-inner', type=int, required=True,
                        help='Number of inner conditional samples.')
    parser.add_argument('--randomize', type=bool, required=False, default=True,
                        help='Let the samples be random after permutation sequence fixing.')
    return parser


def cli_action(args):
    """Run sampling method

    Parameters
    ----------
    args : argparse namespace
    """
    problem = read_param_file(args.paramfile)
    param_values = sample(problem, 
                          args.n_perms, 
                          args.samples, 
                          args.n_outer, 
                          args.n_inner, 
                          args.randomize)

    np.savetxt(args.output, param_values, delimiter=args.delimiter,
               fmt='%.' + str(args.precision) + 'e')