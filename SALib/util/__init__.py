__all__ = ["scale_samples", "read_param_file"]
import csv
import numpy as np

# Rescale samples from [0, 1] to [lower, upper]
def scale_samples(params, bounds):
    '''
    Rescales samples in 0-to-1 range to arbitrary bounds.

    Arguments:
        bounds - list of lists of dimensions num_params-by-2
        params - numpy array of dimensions num_params-by-N, where N is the number
        of samples
    '''
    # Check bounds are legal (upper bound is greater than lower bound)
    b = np.array(bounds)
    lower_bounds = b[:,0]
    upper_bounds = b[:,1]

    if np.any(lower_bounds >= upper_bounds):
        raise ValueError("Bounds are not legal")

    # This scales the samples in-place, by using the optional output
    # argument for the numpy ufunctions
    # The calculation is equivalent to:
    #   sample * (upper_bound - lower_bound) + lower_bound
    np.add(np.multiply(params,
                      (upper_bounds - lower_bounds),
                      out=params), \
           lower_bounds,
           out=params)


def read_param_file(filename):


    names = []
    bounds = []
    num_vars = 0

    with open(filename) as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(16384))
        csvfile.seek(0)
        reader = csv.reader(csvfile, dialect)
        for row in reader:
            if row[0].strip().startswith('#'):
                pass
            else:
                num_vars += 1
                names.append(row[0])
                bounds.append([float(row[1]), float(row[2])])

    return {'names': names, 'bounds': bounds, 'num_vars': num_vars}


def read_group_file(filename):
    '''
    Reads in a group file and returns a dictionary containing
        a list of names of the variables
        a numpy matrix of factor group membership
    '''
    output = []

    with open(filename) as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(16384))
        csvfile.seek(0)
        reader = csv.reader(csvfile, dialect)
        for row in reader:
          if row[0].strip().startswith('#'):
              pass
          else:
              output.append([row[0], row[1:]])
    return {'groups': output}

