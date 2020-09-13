import pkgutil
import csv
from warnings import warn


def avail_approaches(pkg):
    '''Create list of available modules.

    Parameters
    ----------
    pkg : module
        module to inspect

    Returns
    ---------
    method : list
        A list of available submodules
    '''
    methods = [modname for importer, modname, ispkg in
               pkgutil.walk_packages(path=pkg.__path__)
               if modname not in
               ['common_args', 'directions', 'sobol_sequence']]
    return methods


def read_param_file(filename, delimiter=None):
    """Unpacks a parameter file into a dictionary

    Reads a parameter file of format::

        Param1,0,1,Group1,dist1
        Param2,0,1,Group2,dist2
        Param3,0,1,Group3,dist3

    (Group and Dist columns are optional)

    Returns a dictionary containing:
        - names - the names of the parameters
        - bounds - a list of lists of lower and upper bounds
        - num_vars - a scalar indicating the number of variables
                     (the length of names)
        - groups - a list of group names (strings) for each variable
        - dists - a list of distributions for the problem,
                    None if not specified or all uniform

    Parameters
    ----------
    filename : str
        The path to the parameter file
    delimiter : str, default=None
        The delimiter used in the file to distinguish between columns

    """
    names = []
    bounds = []
    groups = []
    dists = []
    num_vars = 0
    fieldnames = ['name', 'lower_bound', 'upper_bound', 'group', 'dist']

    with open(filename, 'r') as csvfile:
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
                    groups.append(row['name'])
                elif row['group'] is 'NA':
                    groups.append(row['name'])
                else:
                    groups.append(row['group'])

                # If the fifth column does not contain a distribution
                # use uniform
                if row['dist'] is None:
                    dists.append('unif')
                else:
                    dists.append(row['dist'])

    if groups == names:
        groups = None
    elif len(set(groups)) == 1:
        raise ValueError('''Only one group defined, results will not be
            meaningful''')

    # setting dists to none if all are uniform
    # because non-uniform scaling is not needed
    if all([d == 'unif' for d in dists]):
        dists = None

    return {'names': names, 'bounds': bounds, 'num_vars': num_vars,
            'groups': groups, 'dists': dists}


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