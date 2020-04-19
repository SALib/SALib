from __future__ import division
from __future__ import print_function

from scipy.stats import norm

import numpy as np

from . import common_args
from ..util import read_param_file, compute_groups_matrix, ResultDict
from ..sample.morris import compute_delta


def analyze(problem, X, Y,
            num_resamples=100,
            conf_level=0.95,
            print_to_console=False,
            num_levels=4,
            seed=None):
    """Perform Morris Analysis on model outputs.

    Returns a dictionary with keys 'mu', 'mu_star', 'sigma', and
    'mu_star_conf', where each entry is a list of parameters containing
    the indices in the same order as the parameter file.

    Arguments
    ---------
    problem : dict
        The problem definition
    X : numpy.matrix
        The NumPy matrix containing the model inputs of dtype=float
    Y : numpy.array
        The NumPy array containing the model outputs of dtype=float
    num_resamples : int
        The number of resamples used to compute the confidence
        intervals (default 1000)
    conf_level : float
        The confidence interval level (default 0.95)
    print_to_console : bool
        Print results directly to console (default False)
    num_levels : int
        The number of grid levels, must be identical to the value
        passed to SALib.sample.morris (default 4)

    Returns
    -------
    Si : dict
        A dictionary of sensitivity indices containing the following entries.

        - `mu` - the mean elementary effect
        - `mu_star` - the absolute of the mean elementary effect
        - `sigma` - the standard deviation of the elementary effect
        - `mu_star_conf` - the bootstrapped confidence interval
        - `names` - the names of the parameters

    References
    ----------
    .. [1] Morris, M. (1991).  "Factorial Sampling Plans for Preliminary
           Computational Experiments."  Technometrics, 33(2):161-174,
           doi:10.1080/00401706.1991.10484804.
    .. [2] Campolongo, F., J. Cariboni, and A. Saltelli (2007).  "An effective
           screening design for sensitivity analysis of large models."
           Environmental Modelling & Software, 22(10):1509-1518,
           doi:10.1016/j.envsoft.2006.10.004.

    Examples
    --------
    >>> X = morris.sample(problem, 1000, num_levels=4)
    >>> Y = Ishigami.evaluate(X)
    >>> Si = morris.analyze(problem, X, Y, conf_level=0.95,
    >>>                     print_to_console=True, num_levels=4)

    """
    if seed:
        np.random.seed(seed)

    msg = ("dtype of {} array must be 'float', float32 or float64")
    if X.dtype not in ['float', 'float32', 'float64']:
        raise ValueError(msg.format('X'))
    if Y.dtype not in ['float', 'float32', 'float64']:
        raise ValueError(msg.format('Y'))

    # Assume that there are no groups
    groups = None
    delta = compute_delta(num_levels)

    num_vars = problem['num_vars']

    if (problem.get('groups') is None) & (Y.size % (num_vars + 1) == 0):
        num_trajectories = int(Y.size / (num_vars + 1))
    elif problem.get('groups') is not None:
        groups, unique_group_names = compute_groups_matrix(
            problem['groups'])
        number_of_groups = len(unique_group_names)
        num_trajectories = int(Y.size / (number_of_groups + 1))
    else:
        raise ValueError("Number of samples in model output file must be"
                         "a multiple of (D+1), where D is the number of"
                         "parameters (or groups) in your parameter file.")
    ee = np.zeros((num_vars, num_trajectories))
    ee = compute_elementary_effects(
        X, Y, int(Y.size / num_trajectories), delta)

    # Output the Mu, Mu*, and Sigma Values. Also return them in case this is
    # being called from Python
    Si = ResultDict((k, [None] * num_vars)
                    for k in ['names', 'mu', 'mu_star', 'sigma', 'mu_star_conf'])
    Si['mu'] = np.average(ee, 1)
    Si['mu_star'] = np.average(np.abs(ee), 1)
    Si['sigma'] = np.std(ee, axis=1, ddof=1)
    Si['names'] = problem['names']

    for j in range(num_vars):
        Si['mu_star_conf'][j] = compute_mu_star_confidence(
            ee[j, :], num_trajectories, num_resamples, conf_level)

    if groups is not None:
        # if there are groups, then the elementary effects returned need to be
        # computed over the groups of variables,
        # rather than the individual variables
        Si_grouped = ResultDict((k, [None] * num_vars)
                          for k in ['mu_star', 'mu_star_conf'])
        Si_grouped['mu_star'] = compute_grouped_metric(Si['mu_star'], groups)
        Si_grouped['mu_star_conf'] = compute_grouped_metric(Si['mu_star_conf'],
                                                            groups)
        Si_grouped['names'] = unique_group_names
        Si_grouped['sigma'] = compute_grouped_sigma(Si['sigma'], groups)
        Si_grouped['mu'] = compute_grouped_sigma(Si['mu'], groups)

        Si = Si_grouped
    
    if print_to_console:
        print(Si.to_df().to_string())

    return Si


def compute_grouped_sigma(ungrouped_sigma, group_matrix):
    '''
    Returns sigma for the groups of parameter values in the
    argument ungrouped_metric where the group consists of no more than
    one parameter
    '''

    group_matrix = np.array(group_matrix, dtype=np.bool)

    sigma_masked = np.ma.masked_array(ungrouped_sigma * group_matrix.T,
                                      mask=(group_matrix ^ 1).T)
    sigma_agg = np.ma.mean(sigma_masked, axis=1)
    sigma = np.zeros(group_matrix.shape[1], dtype=np.float)
    np.copyto(sigma, sigma_agg, where=group_matrix.sum(axis=0) == 1)
    np.copyto(sigma, np.NAN, where=group_matrix.sum(axis=0) != 1)

    return sigma


def compute_grouped_metric(ungrouped_metric, group_matrix):
    '''
    Computes the mean value for the groups of parameter values in the
    argument ungrouped_metric
    '''

    group_matrix = np.array(group_matrix, dtype=np.bool)

    mu_star_masked = np.ma.masked_array(ungrouped_metric * group_matrix.T,
                                        mask=(group_matrix ^ 1).T)
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


def compute_elementary_effects(model_inputs, model_outputs, trajectory_size,
                               delta):
    '''
    Arguments
    ---------
    model_inputs : matrix of inputs to the model under analysis.
        x-by-r where x is the number of variables and
        r is the number of rows (a function of x and num_trajectories)
    model_outputs
        an r-length vector of model outputs
    trajectory_size
        a scalar indicating the number of rows in a trajectory
    delta : float
        scaling factor computed from `num_levels`

    Returns
    ---------
    ee : np.array
        Elementary Effects for each parameter
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


def compute_mu_star_confidence(ee, num_trajectories, num_resamples,
                               conf_level):
    '''
    Uses bootstrapping where the elementary effects are resampled with
    replacement to produce a histogram of resampled mu_star metrics.
    This resample is used to produce a confidence interval.
    '''
    if not 0 < conf_level < 1:
        raise ValueError("Confidence level must be between 0-1.")

    resample_index = np.random.randint(
        len(ee), size=(num_resamples, num_trajectories))
    ee_resampled = ee[resample_index]

    # Compute average of the absolute values over each of the resamples
    mu_star_resampled = np.average(np.abs(ee_resampled), axis=1)

    return norm.ppf(0.5 + conf_level / 2) * mu_star_resampled.std(ddof=1)


def cli_parse(parser):
    parser.add_argument('-X', '--model-input-file', type=str,
                        required=True, default=None,
                        help='Model input file')
    parser.add_argument('-r', '--resamples', type=int, required=False,
                        default=1000,
                        help='Number of bootstrap resamples for Sobol \
                           confidence intervals')
    parser.add_argument('-l', '--levels', type=int, required=False,
                        default=4, help='Number of grid levels \
                           (Morris only)')
    parser.add_argument('--grid-jump', type=int, required=False,
                        default=2, help='Grid jump size (Morris only)')
    return parser


def cli_action(args):
    problem = read_param_file(args.paramfile)
    Y = np.loadtxt(args.model_output_file,
                   delimiter=args.delimiter, usecols=(args.column,))
    X = np.loadtxt(args.model_input_file, delimiter=args.delimiter, ndmin=2)
    if len(X.shape) == 1:
        X = X.reshape((len(X), 1))

    analyze(problem, X, Y, num_resamples=args.resamples, print_to_console=True,
            num_levels=args.levels, seed=args.seed)


if __name__ == "__main__":
    common_args.run_cli(cli_parse, cli_action)
