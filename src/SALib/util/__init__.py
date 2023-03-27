"""A set of utility functions

"""
from collections import OrderedDict
from typing import Dict, Tuple
import warnings

import numpy as np  # type: ignore
import scipy as sp  # type: ignore
from typing import List

from .util_funcs import (  # noqa: F401, E402
    avail_approaches,
    read_param_file,
    _check_bounds,
    _check_groups,
)  # noqa: F401, E402
from .problem import ProblemSpec  # noqa: F401, E402
from .results import ResultDict  # noqa: F401, E402


__all__ = ["scale_samples", "read_param_file", "avail_approaches"]


def _scale_samples(params: np.ndarray, bounds: List):
    """Rescale samples in 0-to-1 range to arbitrary bounds

    Parameters
    ----------
    params : numpy.ndarray
        numpy array of dimensions `num_params`-by-:math:`N`,
        where :math:`N` is the number of samples

    bounds : list
        list of lists of dimensions `num_params`-by-2
    """
    # Check bounds are legal (upper bound is greater than lower bound)
    lower_bounds, upper_bounds = _check_bounds(bounds)

    if np.any(lower_bounds >= upper_bounds):
        raise ValueError(
            "Bounds are not legal (upper bound must be greater than lower bound)"
        )

    # This scales the samples in-place, by using the optional output
    # argument for the numpy ufunctions
    # The calculation is equivalent to:
    #   sample * (upper_bound - lower_bound) + lower_bound
    np.add(
        np.multiply(params, (upper_bounds - lower_bounds), out=params),
        lower_bounds,
        out=params,
    )


def scale_samples(params: np.ndarray, problem: Dict):
    """Scale samples based on specified distribution (defaulting to uniform).

    Adds an entry to the problem specification to indicate samples have been
    scaled to maintain backwards compatibility (`sample_scaled`).

    Parameters
    ----------
    params : np.ndarray,
        numpy array of dimensions `num_params`-by-:math:`N`,
        where :math:`N` is the number of samples
    problem : dictionary,
        SALib problem specification

    Returns
    -------
    np.ndarray, scaled samples
    """
    bounds = problem["bounds"]
    dists = problem.get("dists")

    if dists is None:
        _scale_samples(params, bounds)
    else:
        if params.shape[1] != len(dists):
            msg = "Mismatch in number of parameters and distributions.\n"
            msg += "Num parameters: {}".format(params.shape[1])
            msg += "Num distributions: {}".format(len(dists))
            raise ValueError(msg)

        params = _nonuniform_scale_samples(params, bounds, dists)

    problem["sample_scaled"] = True

    return params
    # limited_params = limit_samples(params, upper_bound, lower_bound, dists)


def _unscale_samples(params, bounds):
    """Rescale samples from arbitrary bounds back to [0,1] range

    Parameters
    ----------
    bounds : list
        list of lists of dimensions num_params-by-2
    params : numpy.ndarray
        numpy array of dimensions num_params-by-N,
        where N is the number of samples
    """
    # Check bounds are legal (upper bound is greater than lower bound)
    b = np.array(bounds)
    lower_bounds = b[:, 0]
    upper_bounds = b[:, 1]

    if np.any(lower_bounds >= upper_bounds):
        raise ValueError(
            "Bounds are not legal (upper bound must be greater than lower bound)"
        )

    # This scales the samples in-place, by using the optional output
    # argument for the numpy ufunctions
    # The calculation is equivalent to:
    #   (sample - lower_bound) / (upper_bound - lower_bound)
    np.divide(
        np.subtract(params, lower_bounds, out=params),
        np.subtract(upper_bounds, lower_bounds),
        out=params,
    )


def _nonuniform_scale_samples(params, bounds, dists):
    """Rescale samples in 0-to-1 range to other distributions

    Parameters
    ----------
    params : numpy.ndarray
        numpy array of dimensions num_params-by-N,
        where N is the number of samples
    dists : list
        list of distributions, one for each parameter
            unif: uniform with lower and upper bounds
            triang: triangular with lower and upper bounds, as well as
                    location of peak
                    The location of peak is in percentage of width
                    e.g. :code:`[1.0, 3.0, 0.5]` indicates 1.0 to 3.0 with a
                    peak at 2.0

                    A soon-to-be deprecated two-value format assumes the lower
                    bound to be 0
                    e.g. :code:`[3, 0.5]` assumes 0 to 3, with a peak at 1.5
            norm: normal distribution with mean and standard deviation
            truncnorm: truncated normal distribution with upper and lower
                    bounds, mean and standard deviation
            lognorm: lognormal with ln-space mean and standard deviation
    """
    b = np.array(bounds, dtype=object)

    # initializing matrix for converted values
    conv_params = np.empty_like(params)

    # loop over the parameters
    for i in range(conv_params.shape[1]):
        # setting first and second arguments for distributions
        b1 = b[i][0]  # ending
        b2 = b[i][1]  # 0-1

        if dists[i] == "triang":
            if len(b[i]) == 3:
                loc_start = b[i][0]  # loc start
                b1 = b[i][1]  # triangular distribution end
                b2 = b[i][2]  # 0-1 aka c (the peak)
            elif len(b[i]) == 2:
                msg = (
                    "Two-value format for triangular distributions detected.\n"
                    "To remove this message, specify the distribution start, "
                    "end, and peak (three values) "
                    "instead of the current two-value format "
                    "(distribution end and peak, with start assumed to be 0)\n"
                    "The two-value format will be deprecated in SALib v1.5"
                )
                warnings.warn(msg, DeprecationWarning, stacklevel=2)

                loc_start = 0
                b1 = b[i][0]
                b2 = b[i][1]
            else:
                raise ValueError(
                    "Unknown triangular distribution specification. Check"
                    " problem specification."
                )

            # checking for correct parameters
            if b1 < 0 or b2 < 0 or b2 >= 1 or loc_start > b1:
                raise ValueError(
                    """Triangular distribution bound error: Scale must be
                    greater than zero; peak on interval [0,1], triangular
                    start value must be smaller than end value"""
                )
            else:
                conv_params[:, i] = sp.stats.triang.ppf(
                    params[:, i], c=b2, scale=b1 - loc_start, loc=loc_start
                )

        elif dists[i] == "unif":
            if b1 >= b2:
                raise ValueError(
                    """Uniform distribution: lower bound
                    must be less than upper bound"""
                )
            else:
                conv_params[:, i] = params[:, i] * (b2 - b1) + b1

        elif dists[i] == "norm":
            if b2 <= 0:
                raise ValueError("""Normal distribution: stdev must be > 0""")
            else:
                conv_params[:, i] = sp.stats.norm.ppf(params[:, i], loc=b1, scale=b2)

        # Truncated normal distribution
        # parameters are lower bound and upper bound, mean and stdev
        elif dists[i] == "truncnorm":
            b3 = b[i][2]
            b4 = b[i][3]
            if b4 <= 0:
                raise ValueError(
                    """Truncated normal distribution: stdev must
                    be > 0"""
                )
            if b1 >= b2:
                raise ValueError(
                    """Truncated normal distribution: lower bound
                    must be less than upper bound"""
                )
            else:
                conv_params[:, i] = sp.stats.truncnorm.ppf(
                    params[:, i], (b1 - b3) / b4, (b2 - b3) / b4, loc=b3, scale=b4
                )

        # lognormal distribution (ln-space, not base-10)
        # parameters are ln-space mean and standard deviation
        elif dists[i] == "lognorm":
            # checking for valid parameters
            if b2 <= 0:
                raise ValueError("""Lognormal distribution: stdev must be > 0""")
            else:
                conv_params[:, i] = np.exp(
                    sp.stats.norm.ppf(params[:, i], loc=b1, scale=b2)
                )

        else:
            valid_dists = ["unif", "triang", "norm", "truncnorm", "lognorm"]
            raise ValueError("Distributions: choose one of %s" % ", ".join(valid_dists))

    return conv_params


def extract_group_names(groups: List) -> Tuple:
    """Get a unique set of the group names.

    Reverts to parameter names (and number of parameters) if groups not
    defined.

    Parameters
    ----------
    groups : List


    Returns
    -------
    tuple : names, number of groups
    """
    names = list(OrderedDict.fromkeys(groups))
    number = len(names)

    return names, number


def compute_groups_matrix(groups: List):
    """Generate matrix which notes factor membership of groups

    Computes a k-by-g matrix which notes factor membership of groups
    where:
        k is the number of variables (factors)
        g is the number of groups
    Also returns a g-length list of unique group_names whose positions
    correspond to the order of groups in the k-by-g matrix

    Parameters
    ----------
    groups : List
        Group names corresponding to each variable

    Returns
    -------
    tuple
        containing group matrix assigning parameters to
        groups and a list of unique group names
    """
    num_vars = len(groups)
    unique_group_names, number_of_groups = extract_group_names(groups)

    indices = dict([(x, i) for (i, x) in enumerate(unique_group_names)])

    output = np.zeros((num_vars, number_of_groups), dtype=int)

    for parameter_row, group_membership in enumerate(groups):
        group_index = indices[group_membership]
        output[parameter_row, group_index] = 1

    return output, unique_group_names


def _define_problem_with_groups(problem: Dict) -> Dict:
    """
    Checks if the user defined the 'groups' key in the problem dictionary.
    If not, makes the 'groups' key equal to the variables names. In other
    words, the number of groups will be equal to the number of variables, which
    is equivalent to no groups.

    Parameters
    ----------
    problem : dict
        The problem definition

    Returns
    -------
    problem : dict
        The problem definition with the 'groups' key, even if the user doesn't
        define it
    """
    # Checks if there isn't a key 'groups' or if it exists and is set to 'None'
    if "groups" not in problem or not problem["groups"]:
        problem["groups"] = problem["names"]
    elif len(problem["groups"]) != problem["num_vars"]:
        raise ValueError(
            "Number of entries in 'groups' should be the same " "as in 'names'"
        )
    return problem


def _compute_delta(num_levels: int) -> float:
    """Computes the delta value from number of levels

    Parameters
    ---------
    num_levels : int
        The number of levels

    Returns
    -------
    float
    """
    return num_levels / (2.0 * (num_levels - 1))
