__all__ = ["scale_samples", "read_param_file"]
from collections import OrderedDict
import csv
from warnings import warn

import numpy as np
import scipy as sp

def scale_samples(params, bounds, dists=None):
    '''
    Rescales samples in 0-to-1 range to arbitrary bounds.

    Arguments:
        bounds - list of lists of dimensions num_params-by-2
        params - numpy array of dimensions num_params-by-N,
        where N is the number of samples
        dists - vector of distribution names, None implies uniform
    '''
    if dists == None:
        # computations when only uniform distributoins are used
		
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
	
    else:
    	# computations for non-uniform distributions
    	
        # initializing matrix to hold converted values
        conv_param =  np.empty([len(params),len(params[0])])
		
        if len(dists) != len(params[0]):
            print('Invalid number of distributions')
			
        for i in range(len(dists)):
            # loop over variables and converting them one by one into specified distributions
            
            # uniform distribution
            if dists[i] == 'unif':
                arg1 = bounds[i][0] # lower bound
                arg2 = bounds[i][1] # upper bound
                
                # printing error when upper bound is less than lower bound
                if arg1 > arg2:
                    print('Upper bound must be less than lower bound')
                else:
                    conv_param[:,i] = params[:,i]*(arg2-arg1) + arg1
					
			# normal distribution
            elif dists[i] == 'norm':
                arg1 = bounds[i][0] # mean
                arg2 = bounds[i][1] # standard deviation
                
                # printing error when standard deviation is not positive
                if arg2 <= 0:
                    print('Standard deviation must be positive')
                else:
                    conv_param[:,i] = sp.stats.norm.ppf(params[:,i],loc=arg1,scale=arg2)
					
			# log-normal distribution (base-e not base-10)
            elif dists[i] == 'lognorm':
                arg1 = bounds[i][0] # ln-space mean
                arg2 = bounds[i][1] # ln-space standard deviation
                
                # checking if real-space lower bound is specified
                if(len(bound[i]) == 3):
                    arg3 = bounds[i][2]
                else:
                    arg3 = 0. # lower bound
                
                # printing error if log-space standard deviation is not positive
                if arg2[i] <= 0:
                    print('Standard deviation must be positive')
                else:
                    conv_param[:,i] = exp(sp.stats.norm.ppf(params[:,i],loc=arg1,scale=arg2))+arg3
			
			# triangular distribution
            elif dists[i] == 'triang':
                arg1 = bounds[i][0] # lower bound
                arg2 = bounds[i][1] # upper bound
                
                # condition is most-likely value specified
                # if not specified, use average of upper and lower bounds
                if(len(bound[i]) == 3):
                    arg3 = bounds[i][2] # most likely
                else:
                    arg3 = np.mean([arg1,arg2]) # lower bound
                
                # checking upper bound less than lower bound
                # and that most likely value within the bounds
                if arg1 > arg2:
                    print('Upper bound must be greater than lower bound')
                elif (arg3 < arg1) or (arg3 > arg2):
                    print('Most likely value must be within upper and lower bounds')
                else:
                    a = arg1 # location (lower bound)
                    b = arg2 - arg1 # scale (width of distribution)
                    c = (arg3-a)/b # most likely as percentage of scale
                    conv_param[:,i] = sp.stats.triang.ppf(params[:,i],c=c,loc=a,scale=b)
            
            # error message for a distribution type that is not valid
            else:
                print('Not a valid distribution type')
    	return(conv_param)

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

def read_param_file(filename, delimiter=None):
    '''
    Reads a parameter file of format:
        Param1,0,1,Group1
        Param2,0,1,Group2
        Param3,0,1,Group3
    And returns a dictionary containing:
        - names - the names of the parameters
        - bounds - a list of lists of lower and upper bounds
        - num_vars - a scalar indicating the number of variables
                     (the length of names)
        - groups - a tuple containing i) a group matrix assigning parameters to
                   groups
                                      ii) a list of unique group names
    '''
    names = []
    bounds = []
    group_list = []
    num_vars = 0
    fieldnames = ['name', 'lower_bound', 'upper_bound', 'group']

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
                else:
                    group_list.append(row['group'])

        group_matrix, group_names = compute_groups_from_parameter_file(
            group_list, num_vars)

        if np.all(group_matrix == np.eye(num_vars)):
            group_tuple = None
        else:
            group_tuple = (group_matrix, group_names)

    return {'names': names, 'bounds': bounds, 'num_vars': num_vars,
            'groups': group_tuple}


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
