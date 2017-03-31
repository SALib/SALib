from __future__ import division

from . sobol_lib import i4_sobol_generate


def sample(n, d):
    """Generate a numpy array of Sobol sequence samples

    Parameters
    ----------
    n : int
        the number of points to generate
    d : int
        the spatial dimension

    Returns
    -------
    :class:`numpy.array`
    """

    result = i4_sobol_generate(d, n, 0)
    return result.T
