from __future__ import division
from __future__ import print_function
from ..util import read_param_file
from sys import exit
import numpy as np
from scipy.stats import norm
from . import common_args

# Perform Morris Analysis on file of model results
# Returns a dictionary with keys 'mu', 'mu_star', 'sigma', and 'mu_star_conf'
# Where each entry is a list of size num_vars (the number of parameters)
# Containing the indices in the same order as the parameter file


def analyze(pfile,
            input_file,
            output_file,
            column=0,
            delim=' ',
            num_resamples=1000,
            conf_level=0.95,
            print_to_console=False,
            grid_jump=2,
            num_levels=4):

    # Assume that there are no groups
    groups = None

    delta = grid_jump / (num_levels - 1)

    param_file = read_param_file(pfile,True)
    Y = np.loadtxt(output_file, delimiter=delim, usecols=(column,))
    X = np.loadtxt(input_file, delimiter=delim, ndmin=2)
    if len(X.shape) == 1:
        X = X.reshape((len(X), 1))

    num_vars = param_file['num_vars']

    if (Y.size % (num_vars + 1) == 0):
        num_trajectories = int(Y.size / (num_vars + 1))
    elif param_file['groups'] is not None:
        groups, unique_group_names = param_file['groups']
        number_of_groups = len(unique_group_names)
        num_trajectories = int(Y.size / (number_of_groups + 1))
    else:
        raise ValueError("""Number of samples in model output file must be a multiple of (D+1), \
                            where D is the number of parameters in your parameter file. \
                         """)
    ee = np.zeros((num_vars, num_trajectories))
    ee = compute_effects_vector(X, Y, Y.size / num_trajectories, delta)

    # Output the Mu, Mu*, and Sigma Values. Also return them in case this is
    # being called from Python
    Si = dict((k, [None] * num_vars)
              for k in ['names', 'mu', 'mu_star', 'sigma', 'mu_star_conf'])
    if print_to_console:
        print("Parameter Mu Sigma Mu_Star Mu_Star_Conf")

    Si['mu'] = np.average(ee, 1)
    Si['mu_star'] = np.average(np.abs(ee), 1)
    Si['sigma'] = np.std(ee, 1)

    Si['names'] = param_file['names']

    for j in range(num_vars):
        Si['mu_star_conf'][j] = compute_mu_star_confidence(
            ee[j, :], num_trajectories, num_resamples, conf_level)

        if print_to_console:
            print("%s %f %f %f %f" % (param_file['names'][j], Si['mu'][j], Si[
                  'sigma'][j], Si['mu_star'][j], Si['mu_star_conf'][j]))

    if groups is None:
        return Si
    elif groups is not None:
        # if there are groups, then the elementary effects returned need to be
        # computed over the groups of variables, rather than the individual variables
        Si_grouped = dict((k, [None] * num_vars)
                for k in ['mu_star', 'mu_star_conf'])
        Si_grouped['mu_star'] = compute_grouped_mu_star(Si['mu_star'], groups)
        Si_grouped['mu_star_conf'] = compute_grouped_mu_star(Si['mu_star_conf'], groups)
        Si_grouped['names'] = unique_group_names
        return Si_grouped
    else:
        raise RuntimeError("Could determine which parameters should be returned")


def compute_grouped_mu_star(mu_star_ungrouped, group_matrix):

    group_matrix = np.array(group_matrix)
    mu_star_grouped = np.divide(np.dot(mu_star_ungrouped, group_matrix), np.sum(group_matrix, 0))

    return mu_star_grouped.T


def compute_elementary_effect(X, Y, j1, j2):
    # The elementary effect is (change in output)/(change in input)
    # Each parameter has one EE per trajectory, because it is only changed
    # once in each trajectory
    try:

        change_in_input = X[j2, :] - X[j1, :]
        change_in_output = Y[j2] - Y[j1]
        return np.linalg.solve(change_in_input, change_in_output)

    except np.linalg.linalg.LinAlgError:

        ee = np.zeros(X.shape[1])
        for k in j1:
            delta = X[k+1, :] - X[k, :]
            col = np.nonzero(delta)[0]
            ee[k] = np.array((Y[k + 1] - Y[k]) / (X[k + 1, col] - X[k, col]))
        return ee

def get_increased_values(op_vec, up, lo):

    up = np.pad(up, ((0, 0), (1, 0), (0, 0)), 'constant')
    lo = np.pad(lo, ((0, 0), (0, 1), (0, 0)), 'constant')

    res = np.einsum('ik,ikj->ij', op_vec, up+lo)

    return res.T


def get_decreased_values(op_vec, up, lo):

    up = np.pad(up, ((0, 0), (0, 1), (0, 0)), 'constant')
    lo = np.pad(lo, ((0, 0), (1, 0), (0, 0)), 'constant')

    res = np.einsum('ik,ikj->ij', op_vec, up+lo)

    return res.T


def compute_effects_vector(model_inputs, model_outputs, trajectory_size, delta):
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

    ip_vec = model_inputs.reshape(num_trajectories,trajectory_size,num_vars)
    ip_cha = np.subtract(ip_vec[:,1:,:], ip_vec[:,0:-1,:])
    up = (ip_cha > 0)
    lo = (ip_cha < 0)

    op_vec = model_outputs.reshape(num_trajectories,trajectory_size)

    result_up = get_increased_values(op_vec, up, lo)
    result_lo = get_decreased_values(op_vec, up, lo)

    ee = np.subtract(result_up, result_lo)
    np.divide(ee, delta, out = ee)

    return ee


def compute_mu_star_confidence(ee, num_trajectories, num_resamples, conf_level):

    ee_resampled = np.empty([num_trajectories])
    mu_star_resampled = np.empty([num_resamples])

    if conf_level < 0 or conf_level > 1:
        raise ValueError("Confidence level must be between 0-1.")

    for i in range(num_resamples):
        for j in range(num_trajectories):

            index = np.random.randint(0, num_trajectories)
            ee_resampled[j] = ee[index]

        mu_star_resampled[i] = np.average(np.abs(ee_resampled))

    return norm.ppf(0.5 + conf_level / 2) * mu_star_resampled.std(ddof=1)

if __name__ == "__main__":

    parser = common_args.create()
    parser.add_argument('-X', '--model-input-file', type=str,
                        required=True, default=None, help='Model input file')
    parser.add_argument('-r', '--resamples', type=int, required=False, default=1000,
                        help='Number of bootstrap resamples for Sobol confidence intervals')

    args = parser.parse_args()
    analyze(args.paramfile, args.model_input_file, args.model_output_file, args.column,
            delim=args.delimiter, num_resamples=args.resamples, print_to_console=True)
