from __future__ import division
from __future__ import print_function

from scipy.stats import norm

import numpy as np

from . import common_args
from ..util import read_param_file


# Perform Morris Analysis on file of model results
# Returns a dictionary with keys 'mu', 'mu_star', 'sigma', and 'mu_star_conf'
# Where each entry is a list of size num_vars (the number of parameters)
# Containing the indices in the same order as the parameter file
def analyze(problem, X, Y,
            num_resamples=1000,
            conf_level=0.95,
            print_to_console=False,
            grid_jump=2,
            num_levels=4):

    # Assume that there are no groups
    groups = None

    delta = grid_jump / (num_levels - 1)

    num_vars = problem['num_vars']

    if (problem['groups'] is None) & (Y.size % (num_vars + 1) == 0):
        num_trajectories = int(Y.size / (num_vars + 1))
    elif problem['groups'] is not None:
        groups, unique_group_names = problem['groups']
        number_of_groups = len(unique_group_names)
        num_trajectories = int(Y.size / (number_of_groups + 1))
    else:
        raise ValueError("""Number of samples in model output file must be a multiple of (D+1), \
                            where D is the number of parameters (or groups) in your parameter file. \
                         """)
    ee = np.zeros((num_vars, num_trajectories))
    ee = compute_elementary_effects(X, Y, int(Y.size / num_trajectories), delta)

    # Output the Mu, Mu*, and Sigma Values. Also return them in case this is
    # being called from Python
    Si = dict((k, [None] * num_vars)
              for k in ['names', 'mu', 'mu_star', 'sigma', 'mu_star_conf'])
    Si['mu'] = np.average(ee, 1)
    Si['mu_star'] = np.average(np.abs(ee), 1)
    Si['sigma'] = np.std(ee, axis=1, ddof=1)
    Si['names'] = problem['names']

    for j in range(num_vars):
        Si['mu_star_conf'][j] = compute_mu_star_confidence(
            ee[j, :], num_trajectories, num_resamples, conf_level)

    if groups is None:
        if print_to_console:
            print("Parameter Mu Sigma Mu_Star Mu_Star_Conf")
            for j in range(num_vars):
                print("%s %f %f %f %f" % (problem['names'][j], Si['mu'][j], Si[
                    'sigma'][j], Si['mu_star'][j], Si['mu_star_conf'][j]))
        return Si
    elif groups is not None:
        # if there are groups, then the elementary effects returned need to be
        # computed over the groups of variables, rather than the individual variables
        Si_grouped = dict((k, [None] * num_vars)
                for k in ['mu_star', 'mu_star_conf'])
        Si_grouped['mu_star'] = compute_grouped_metric(Si['mu_star'], groups)
        Si_grouped['mu_star_conf'] = compute_grouped_metric(Si['mu_star_conf'],
                                                             groups)
        Si_grouped['names'] = unique_group_names
        Si_grouped['sigma'] = None

        if print_to_console:
            print("Parameter Mu_Star Mu_Star_Conf")
            for j in list(range(number_of_groups)):
                print("%s %f %f" % (Si_grouped['names'][j],
                                    Si_grouped['mu_star'][j],
                                    Si_grouped['mu_star_conf'][j]))

        return Si_grouped
    else:
        raise RuntimeError("Could not determine which parameters should be returned")


def compute_grouped_metric(ungrouped_metric, group_matrix):
    '''
    Computes the mean value for the groups of parameter values in the
    argument ungrouped_metric
    '''

    group_matrix = np.array(group_matrix, dtype=np.bool)

    mu_star_masked = np.ma.masked_array(ungrouped_metric * group_matrix.T, 
                                        mask=(group_matrix^1).T)
    mean_of_mu_star = np.ma.mean(mu_star_masked, axis=1)

    return mean_of_mu_star


def get_increased_values(op_vec, up, lo):

    up = np.pad(up, ((0, 0), (1, 0), (0, 0)), 'constant')
    lo = np.pad(lo, ((0, 0), (0, 1), (0, 0)), 'constant')

    res = np.einsum('ik,ikj->ij', op_vec, up + lo)

    return res.T


def get_decreased_values(op_vec, up, lo):

    up = np.pad(up, ((0, 0), (0, 1), (0, 0)), 'constant')
    lo = np.pad(lo, ((0, 0), (1, 0), (0, 0)), 'constant')

    res = np.einsum('ik,ikj->ij', op_vec, up + lo)

    return res.T


def compute_elementary_effects(model_inputs, model_outputs, trajectory_size, delta):
    '''
    Arguments:
        - model_inputs - matrix of inputs to the model under analysis.
                         x-by-r where x is the number of variables and
                         r is the number of rows (a function of x and num_trajectories)
        - model_outputs - an r-length vector of model outputs
        - trajectory_size - a scalar indicating the number of rows in a
                            trajectory
    '''
    num_vars = model_inputs.shape[1]
    num_rows = model_inputs.shape[0]
    num_trajectories = int(num_rows / trajectory_size)

    ee = np.zeros((num_trajectories, num_vars), dtype=np.float)

    ip_vec = model_inputs.reshape(num_trajectories, trajectory_size, num_vars)
    ip_cha = np.subtract(ip_vec[:, 1:, :], ip_vec[:, 0:-1, :])
    up = (ip_cha > 0)
    lo = (ip_cha < 0)

    op_vec = model_outputs.reshape(num_trajectories, trajectory_size)

    result_up = get_increased_values(op_vec, up, lo)
    result_lo = get_decreased_values(op_vec, up, lo)

    ee = np.subtract(result_up, result_lo)
    np.divide(ee, delta, out=ee)

    return ee


def compute_mu_star_confidence(ee, num_trajectories, num_resamples, conf_level):
    '''
    Uses bootstrapping where the elementary effects are resampled with replacement
    to produce a histogram of resampled mu_star metrics.
    This resample is used to produce a confidence interval.
    '''
    ee_resampled = np.empty([num_trajectories])
    mu_star_resampled = np.empty([num_resamples])

    if conf_level < 0 or conf_level > 1:
        raise ValueError("Confidence level must be between 0-1.")

    resample_index = np.random.randint(len(ee), size=(num_resamples, num_trajectories))
    ee_resampled = ee[resample_index]
    # Compute average of the absolute values over each of the resamples
    mu_star_resampled = np.average(np.abs(ee_resampled), axis=1)

    return norm.ppf(0.5 + conf_level / 2) * mu_star_resampled.std(ddof=1)


if __name__ == "__main__":

    parser = common_args.create()
    parser.add_argument('-X', '--model-input-file', type=str,
                        required=True, default=None, help='Model input file')
    parser.add_argument('-r', '--resamples', type=int, required=False, default=1000,
                        help='Number of bootstrap resamples for Sobol confidence intervals')
    parser.add_argument('-l', '--levels', type=int, required=False,
                        default=4, help='Number of grid levels (Morris only)')
    parser.add_argument('--grid-jump', type=int, required=False,
                        default=2, help='Grid jump size (Morris only)')
    args = parser.parse_args()

    problem = read_param_file(args.paramfile)

    Y = np.loadtxt(args.model_output_file, delimiter=args.delimiter, usecols=(args.column,))
    X = np.loadtxt(args.model_input_file, delimiter=args.delimiter, ndmin=2)
    if len(X.shape) == 1:
        X = X.reshape((len(X), 1))

    analyze(problem, X, Y, num_resamples=args.resamples, print_to_console=True,
            num_levels=args.levels, grid_jump=args.grid_jump)
