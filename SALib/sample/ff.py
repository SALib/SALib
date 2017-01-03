"""The sampling implementation of fractional factorial method

This implementation is based on the formulation put forward in
[`Saltelli et al. 2008 <http://www.wiley.com/WileyCDA/WileyTitle/productCd-0470059974.html>`_]

"""

from scipy.linalg import hadamard
import numpy as np
from . import common_args
from ..util import scale_samples, read_param_file


def find_smallest(num_vars):
    """Find the smallest exponent of two that is greater than the number
    of variables

    Arguments
    =========
    num_vars : int
        Number of variables

    Returns
    =======
    x : int
        Smallest exponent of two greater than `num_vars`
    """
    for x in range(10):
        if num_vars <= 2 ** x:
            return x


def extend_bounds(problem):
    """Extends the problem bounds to the nearest power of two

    Arguments
    =========
    problem : dict
        The problem definition
    """

    num_vars = problem['num_vars']
    num_ff_vars = 2 ** find_smallest(num_vars)
    num_dummy_variables = num_ff_vars - num_vars

    bounds = list(problem['bounds'])
    names = problem['names']
    if num_dummy_variables > 0:
        bounds.extend([[0, 1] for x in range(num_dummy_variables)])
        names.extend(["dummy_" + str(var) for var in range(num_dummy_variables)])
        problem['bounds'] = bounds
        problem['names'] = names
        problem['num_vars'] = num_ff_vars

    return problem

def generate_contrast(problem):
    """Generates the raw sample from the problem file

    Arguments
    =========
    problem : dict
        The problem definition
    """

    num_vars = problem['num_vars']

    # Find the smallest n, such that num_vars < k
    k = [2 ** n for n in range(16)]
    k_chosen = 2 ** find_smallest(num_vars)

    # Generate the fractional factorial contrast
    contrast = np.vstack([hadamard(k_chosen), -hadamard(k_chosen)])

    return contrast

def sample(problem):
    """Generates model inputs using a fractional factorial sample

    Returns a NumPy matrix containing the model inputs required for a
    fractional factorial analysis.
    The resulting matrix has D columns, where D is smallest power of 2 that is
    greater than the number of parameters.
    These model inputs are intended to be used with
    :func:`SALib.analyze.ff.analyze`.

    The problem file is padded with a number of dummy variables called
    ``dummy_0`` required for this procedure. These dummy variables can be used
    as a check for errors in the analyze procedure.

    This algorithm is an implementation of that contained in
    [`Saltelli et al. 2008 <http://www.wiley.com/WileyCDA/WileyTitle/productCd-0470059974.html>`_]

    Arguments
    =========
    problem : dict
        The problem definition

    Returns
    =======
    sample : :class:`numpy.array`

    """

    contrast = generate_contrast(problem)
    sample = np.array((contrast + 1.) / 2, dtype=np.float)
    problem = extend_bounds(problem)
    scale_samples(sample, problem['bounds'])
    return sample

if __name__ == "__main__":

    parser = common_args.create()
    args = parser.parse_args()

    problem = read_param_file(args.paramfile)
    param_values = sample(problem)

    np.savetxt(args.output, param_values, delimiter=args.delimiter,
               fmt='%.' + str(args.precision) + 'e')
