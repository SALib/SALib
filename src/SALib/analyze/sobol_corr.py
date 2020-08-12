import numpy as np
import openturns as ot


def analyze(problem, x, y, n_sample, n_boot=100, estimator='soboleff2'):
    """
    Computes the Sobol' indices with the pick and freeze strategy.
    
    The method computes the indices from the previously created samples.

    Parameters
    ----------            
    n_boot : int, optional (default=500)
        The bootstrap sample size.
        
    estimator : str, optional (default='soboleff2')
        The estimator method for the pick and freeze strategy. Available
        estimators :
            
        - 'sobol': initial pick and freeze from [1],
        - 'sobol2002': from [2],
        - 'sobol2007': from [3],
        - 'soboleff1': first estimator of [4],
        - 'soboleff2': second estimator of [4],
        - 'sobolmara': from [5],
        
    Returns
    -------
    results : SensitivityResults instance
        The computed Sobol' indices.
        
    References
    ----------
    .. [1] Sobol 93
    .. [2] Saltelli & al. 2002
    .. [3] Sobol 2007
    .. [4] Janon
    .. [5] TODO: check the sources
    """
    assert isinstance(n_boot, int), \
        "The number of bootstrap should be an integer"
    assert isinstance(estimator, str), \
        "The estimator name should be an string" 
    assert n_boot > 0, \
        "The number of boostrap should be positive: %d<0" % (n_boot)
    assert estimator in _ESTIMATORS, "Unknow estimator %s" % (estimator)

    assert isinstance(problem, dict), "problem should be of dict type"
    dim = problem['num_vars']
    n_realization = 1

    all_output_sample_1 = np.zeros((dim, n_sample, n_realization))
    all_output_sample_2 = np.zeros((dim, n_sample, n_realization))
    all_output_sample_2t = np.zeros((dim, n_sample, n_realization))
    all_output_sample_2t1 = np.zeros((dim, n_sample, n_realization))

    y = np.array(y)
    for i in range(dim):
        output_sample_i = y[4*n_sample*i:(i+1)*4*n_sample].reshape(4*n_sample, n_realization)
        all_output_sample_1[i] = output_sample_i[:n_sample]
        all_output_sample_2[i] = output_sample_i[n_sample:2*n_sample]
        all_output_sample_2t[i] = output_sample_i[2*n_sample:3*n_sample]
        all_output_sample_2t1[i] = output_sample_i[3*n_sample:]

    results = {}
    for indice_type in ['full', 'ind']:
        first_indices = np.zeros((dim, n_boot, n_realization))
        total_indices = np.zeros((dim, n_boot, n_realization))

        dev = _DELTA_INDICES[indice_type]
        if indice_type == 'full':
            sample_Y2t = all_output_sample_2t
        elif indice_type == 'ind':
            sample_Y2t = all_output_sample_2t1
        else:
            raise ValueError('Unknow type of indice: {0}'.format(indice_type))

        # TODO: cythonize this, takes too much memory when n_boot is large
        boot_idx = None
        for i in range(dim):
            if n_boot > 1:
                boot_idx = np.zeros((n_boot, n_sample), dtype=int)
                boot_idx[0] = range(n_sample)
                boot_idx[1:] = np.random.randint(0, n_sample, size=(n_boot-1, n_sample))

            Y1 = all_output_sample_1[i]
            Y2 = all_output_sample_2[i]
            Y2t = sample_Y2t[i]
            first, total = sobol_indices(Y1, Y2, Y2t, boot_idx=boot_idx, estimator=estimator)
            if first is not None:
                first = first.reshape(n_boot, n_realization)
            if total is not None:
                total = total.reshape(n_boot, n_realization)

            first_indices[i-dev], total_indices[i-dev] = first, total

        if np.isnan(total_indices).all():
            total_indices = None

        results[f'S1_{indice_type}'] = first_indices.reshape(dim, -1).mean(axis=1)
        results[f'ST_{indice_type}'] = total_indices.reshape(dim, -1).mean(axis=1)

    return results



# TODO: cythonize this, it takes too much memory in vectorial
def sobol_indices(Y1, Y2, Y2t, boot_idx=None, estimator='sobol2002'):
    """Compute the Sobol indices from the to

    Parameters
    ----------
    Y1 : array,
        The 

    Returns
    -------
    first_indice : int or array,
        The first order sobol indice estimation.

    total_indice : int or array,
        The total sobol indice estimation.
    """
    n_sample = Y1.shape[0]
    assert n_sample == Y2.shape[0], "Matrices should have the same sizes"
    assert n_sample == Y2t.shape[0], "Matrices should have the same sizes"
    assert estimator in _ESTIMATORS, 'Unknow estimator {0}'.format(estimator)

    estimator = _ESTIMATORS[estimator]

    # When boot_idx is None, it reshapes the Y as (1, -1).
    first_indice, total_indice = estimator(Y1[boot_idx], Y2[boot_idx], Y2t[boot_idx])

    return first_indice, total_indice


m = lambda x : x.mean(axis=1)
s = lambda x : x.sum(axis=1)
v = lambda x : x.var(axis=1)


def sobol_estimator(Y1, Y2, Y2t):
    """
    """
    mean2 = m(Y1)**2
    var = v(Y1)

    var_indiv = m(Y2t * Y1) - mean2
    first_indice = var_indiv / var
    total_indice = None

    return first_indice, total_indice


def sobol2002_estimator(Y1, Y2, Y2t):
    """
    """
    n_sample = Y1.shape[1]
    mean2 = s(Y1*Y2)/(n_sample - 1)
    var = v(Y1)

    var_indiv = s(Y2t * Y1)/(n_sample - 1) - mean2
    var_total = s(Y2t * Y2)/(n_sample - 1) - mean2
    first_indice = var_indiv / var
    total_indice = 1. - var_total  / var

    return first_indice, total_indice


def sobol2007_estimator(Y1, Y2, Y2t):
    """
    """
    var = v(Y1)

    var_indiv = m((Y2t - Y2) * Y1)
    var_total = m((Y2t - Y1) * Y2)
    first_indice = var_indiv / var
    total_indice = 1. - var_total / var

    return first_indice, total_indice


def soboleff1_estimator(Y1, Y2, Y2t):
    """
    """
    mean2 = m(Y1) * m(Y2t)
    var = m(Y1**2) - m(Y1)**2

    var_indiv = m(Y2t * Y1) - mean2
    var_total = m(Y2t * Y2) - mean2

    first_indice = var_indiv / var
    total_indice = 1. - var_total  / var

    return first_indice, total_indice


def soboleff2_estimator(Y1, Y2, Y2t):
    """
    """
    mean2 = m((Y1 + Y2t)/2.)**2
    var = m((Y1**2 + Y2t**2 )/2.) - m((Y1 + Y2t)/2.)**2

    var_indiv = m(Y2t * Y1) - mean2
    var_total = m(Y2t * Y2) - mean2

    first_indice = var_indiv / var
    total_indice = 1. - var_total  / var

    return first_indice, total_indice


def sobolmara_estimator(Y1, Y2, Y2t):
    """
    """
    if True:
        diff = Y2t - Y2
        var = v(Y1)
    
        var_indiv = m(Y1 * diff)
        var_total = m(diff ** 2)
    
        first_indice = var_indiv / var
        total_indice = var_total / var / 2.
    else:
        n_sample, n_boot, n_realization = Y2.shape
        first_indice = np.zeros((n_boot, n_realization))
        total_indice = np.zeros((n_boot, n_realization))
        for i in range(n_realization):
            diff = Y2t[:, :, i] - Y2[:, :, i]
            var = v(Y1[:, :, i])
        
            var_indiv = m(Y1[:, :, i] * diff)
            var_total = m(diff ** 2)
        
            first_indice[:, i] = var_indiv / var
            total_indice[:, i] = var_total / var / 2.

    return first_indice, total_indice


_ESTIMATORS = {
    'sobol': sobol_estimator,
    'sobol2002': sobol2002_estimator,
    'sobol2007': sobol2007_estimator,
    'soboleff1': soboleff1_estimator,
    'soboleff2': soboleff2_estimator,
    'sobolmara': sobolmara_estimator
    }

_DELTA_INDICES = {
        'full': 0,
        'ind': 1,
        }