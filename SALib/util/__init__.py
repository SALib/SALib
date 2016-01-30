__all__ = ["scale_samples", "read_param_file"]
from collections import OrderedDict
import csv
from warnings import warn

import numpy as np
import scipy as sp

def scale_samples(params, bounds):
    '''
    Rescales samples in 0-to-1 range to arbitrary bounds.

    Arguments:
        bounds - list of lists of dimensions num_params-by-2
        params - numpy array of dimensions num_params-by-N,
        where N is the number of samples
    '''
    # Check bounds are legal (upper bound is greater than lower bound)
    b = np.array(bounds)
    lower_bounds = b[:, 0]
    upper_bounds = b[:, 1]

    if np.any(lower_bounds >= upper_bounds):
        raise ValueError("Bounds are not legal")

    # This scales the samples in-place, by using the optional output
    # argument for the numpy ufunctions
    # The calculation is equivalent to:
    #   sample * (upper_bound - lower_bound) + lower_bound
    np.add(np.multiply(params,
                       (upper_bounds - lower_bounds),
                       out=params),
           lower_bounds,
           out=params)

def unscale_samples(params, bounds):
    '''
    Rescales samples from arbitrary bounds back to [0,1] range.

    Arguments:
        bounds - list of lists of dimensions num_params-by-2
        params - numpy array of dimensions num_params-by-N,
        where N is the number of samples
    '''
    # Check bounds are legal (upper bound is greater than lower bound)
    b = np.array(bounds)
    lower_bounds = b[:, 0]
    upper_bounds = b[:, 1]

    if np.any(lower_bounds >= upper_bounds):
        raise ValueError("Bounds are not legal")

    # This scales the samples in-place, by using the optional output
    # argument for the numpy ufunctions
    # The calculation is equivalent to:
    #   (sample - lower_bound) / (upper_bound - lower_bound)
    np.divide(np.subtract(params, lower_bounds, out=params),
              np.subtract(upper_bounds, lower_bounds),
              out=params)

def nonuniform_scale_samples(params, bounds, dists):
    '''
    Rescales samples in 0-to-1 range to other distributions.

    Arguments:
        problem - problem definition including bounds
        params - numpy array of dimensions num_params-by-N,
            where N is the number of samples
        dists-list of distributions, one for each parameter
                unif: uniform with lower and upper bounds
                triang: triangular with width (scale) and location of peak
                        location of peak is in percentage of width
                        lower bound assumed to be zero
                norm: normal distribution with mean and standard deviation
                lognorm: lognormal with ln-space mean and standard deviation
    '''
    b = np.array(bounds)

    if len(params[0]) != len(dists):
        print('Incorrect number of distributions specified')
        print('Original parameters returned')
        return params
    else:
        # initializing matrix for converted values
        conv_params = np.empty([len(params),len(params[0])])

        # loop over the parameters
        for i in range(len(conv_params[0])):
            # setting first and second arguments for distributions
            arg1 = b[i][0]
            arg2 = b[i][1]

            # triangular distribution
            # paramters are width (scale) and location of peak
            # location of peak is relative to scale
            # e.g., 0.25 means peak is 25% of the width distance from zero
            if dists[i] == 'triang':
                # checking for correct parameters
                if arg1 <= 0:
                    print('Scale must be greater than zero')
                    print('Parameter not converted')
                    conv_params[:,i] = params[:,i]
                elif (arg2 <= 0) or (arg2 >= 1):
                    print('Peak must be on interval [0,1]')
                    print('Parameter not converted')
                    conv_params[:,i] = params[:,i]
                else:
                    conv_params[:,i] = sp.stats.triang.ppf(params[:,i],c=arg2,scale=arg1,loc=0)

            # uniform distribution
            # parameters are lower and upper bounds
            elif dists[i] == 'unif':
                # checking that upper bound is greater than lower bound
                if arg1 >= arg2:
                    print('Lower bound greater than upper bound')
                    print('Parameter not converted')
                    conv_params[:,i] = params[:,i]
                else:
                    conv_params[:,i] = params[:,i]*(arg2-arg1) + arg1

            # normal distribution
            # paramters are mean and standard deviation
            elif dists[i] == 'norm':
                # checking for valid parameters
                if arg2 <= 0:
                    print('Scale must be greater than zero')
                    print('Parameter not converted')
                    conv_params[:,i] = params[:,i]
                else:
                    conv_params[:,i] = sp.stats.norm.ppf(params[:,i],loc=arg1,scale=arg2)

            # lognormal distribution (ln-space, not base-10)
            # paramters are ln-space mean and standard deviation
            elif dists[i] == 'lognorm':
                # checking for valid parameters
                if arg2 <= 0:
                    print('Scale must be greater than zero')
                    print('Parameter not converted')
                    conv_params[:,i] = params[:,i]
                else:
                    conv_params[:,i] = np.exp(sp.stats.norm.ppf(params[:,i],loc=arg1,scale=arg2))

            else:
                print('No valid distribution selected')
    return(conv_params)

def read_param_file(filename, delimiter=None):
    '''
    Reads a parameter file of format:
        Param1,0,1,Group1,dist1
        Param2,0,1,Group2,dist2
        Param3,0,1,Group3,dist3
    And returns a dictionary containing:
        - names - the names of the parameters
        - bounds - a list of lists of lower and upper bounds
        - num_vars - a scalar indicating the number of variables
                     (the length of names)
        - groups - a tuple containing i) a group matrix assigning parameters to
                   groups
                                      ii) a list of unique group names
        - dists - a list of distributions for the problem,
                    None if not specified or all uniform
    '''
    names = []
    bounds = []
    group_list = []
    dist_list = []
    num_vars = 0
    fieldnames = ['name', 'lower_bound', 'upper_bound', 'group', 'dist']
    dist_none_count = 0 # used when evaluating if non-uniform distributions are specified

    with open(filename, 'rU') as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(1024), delimiters=delimiter)
        csvfile.seek(0)
        reader = csv.DictReader(
            csvfile, fieldnames=fieldnames, dialect=dialect)
        for row in reader:
            if row['name'].strip().startswith('#'):
                pass
            else:
                num_vars += 1
                names.append(row['name'])
                bounds.append(
                    [float(row['lower_bound']), float(row['upper_bound'])])

                # If the fourth column does not contain a group name, use
                # the parameter name
                if row['group'] is None:
                    group_list.append(row['name'])
                elif row['group'] is 'NA':
                    group_list.append(row['name'])
                else:
                    group_list.append(row['group'])

                # If the fifth column does not contain a distribution
                # use uniform
                if row['dist'] is None:
                    dist_list.append('unif')
                    dist_none_count += 1
                else:
                    dist_list.append(row['dist'])

    group_matrix, group_names = compute_groups_from_parameter_file(
        group_list, num_vars)

    # setting group_tuple to zero if no groups are defined
    # or all groups are 'NA'
    if np.all(group_matrix == np.eye(num_vars)):
        group_tuple = None
    elif len(np.unique(group_list)) == 1:
        group_tuple = None
    else:
        group_tuple = (group_matrix, group_names)

    # setting dist list to none if all are uniform
    # because non-uniform scaling is not needed
    if dist_none_count == num_vars:
        dist_list = None

    return {'names': names, 'bounds': bounds, 'num_vars': num_vars,
            'groups': group_tuple, 'dists': dist_list}


def compute_groups_from_parameter_file(group_list, num_vars):
    '''
    Computes a k-by-g matrix which notes factor membership of groups
    where:
        k is the number of variables (factors)
        g is the number of groups
    Also returns a g-length list of unique group_names whose positions
    correspond to the order of groups in the k-by-g matrix
    '''
    # Get a unique set of the group names
    unique_group_names = list(OrderedDict.fromkeys(group_list))
    number_of_groups = len(unique_group_names)

    indices = dict([(x, i) for (i, x) in enumerate(unique_group_names)])

    output = np.zeros((num_vars, number_of_groups), dtype=np.int)

    for parameter_row, group_membership in enumerate(group_list):
        group_index = indices[group_membership]
        output[parameter_row, group_index] = 1

    return np.matrix(output), unique_group_names


def requires_gurobipy(_has_gurobi):
    '''
    Decorator function which takes a boolean _has_gurobi as an argument.
    Use decorate any functions which require gurobi.
    Raises an import error at runtime if gurobi is not present.
    Note that all runtime errors should be avoided in the working code,
    using brute force options as preference.
    '''
    def _outer_wrapper(wrapped_function):
        def _wrapper(*args, **kwargs):
            if _has_gurobi:
                result = wrapped_function(*args, **kwargs)
            else:
                warn("Gurobi not available", ImportWarning)
                result = None
            return result
        return _wrapper
    return _outer_wrapper
