from typing import Dict, List
import numpy as np
from scipy.stats import norm

from . import common_args
from ..util import (
    read_param_file,
    compute_groups_matrix,
    ResultDict,
    _define_problem_with_groups,
    _compute_delta,
    _check_groups,
)


def analyze(
    problem: Dict,
    X: np.ndarray,
    SX: np.ndarray,
    Y: np.ndarray,
    num_resamples: int = 100,
    conf_level: float = 0.95,
    print_to_console: bool = False,
    num_levels: int = 4,
    seed=None,
) -> Dict:
    """Perform Morris Analysis on model outputs.

    Returns a result set with keys ``mu``, ``mu_star``, ``sigma``, and
    ``mu_star_conf``, where each entry corresponds to the parameters
    defined in the problem spec or parameter file.

    - ``mu`` metric indicates the mean of the distribution
    - ``mu_star`` metric indicates the mean of the distribution of absolute
        values
    - ``sigma`` is the standard deviation of the distribution

    Notes
    -----
    When applied with groups, the ``mu`` metric is less reliable as the effect
    from parameters within a group become averaged out.

    The ``mu_star`` metric avoids this issue as it indicates the mean of the
    absolute values. If the direction of effects is important, Campolongo et
    al., [2] suggest comparing ``mu_star`` with ``mu``. If ``mu`` is low
    and ``mu_star`` is high, then the effects are of different signs.

    ``sigma`` is used as an indicator of interactions between parameters, or
    groups of parameters.

    Compatible with:
        `morris` : :func:`SALib.sample.morris.sample`


    Examples
    --------
        >>> X = morris.sample(problem, 1000, num_levels=4)
        >>> Y = Ishigami.evaluate(X)
        >>> Si = morris.analyze(problem, X, Y, conf_level=0.95,
        >>>                     print_to_console=True, num_levels=4)


    Parameters
    ----------
    problem : dict
        The problem definition
    X : numpy.array
        The NumPy matrix containing the model inputs of dtype=float
    SX: numpy.array
        The NumPy matrix containing the model nominal inputs of dtype=float
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
    seed : int
        Seed to generate a random number

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
    1. Morris, M. (1991).
       Factorial Sampling Plans for Preliminary Computational Experiments.
       Technometrics, 33(2):161-174,
       doi:10.1080/00401706.1991.10484804.

    2. Campolongo, F., J. Cariboni, and A. Saltelli (2007).
       An effective screening design for sensitivity analysis
       of large models.
       Environmental Modelling & Software, 22(10):1509-1518,
       doi:10.1016/j.envsoft.2006.10.004.

    3. Sin and Gearney (2009)
       Improving the Morris Method for Sensitivity Analysis 
       by Scaling the Elementary Effects.
       19th European Symposium on Computer Aided Process Engineering ESCAPE19:925-930
    """
    if seed:
        np.random.seed(seed)

    _define_problem_with_groups(problem)

    _check_if_array_of_floats(X)
    _check_if_array_of_floats(Y)

    delta = _compute_delta(num_levels)

    num_vars = problem["num_vars"]
    groups = _check_groups(problem)
    if not groups:
        number_of_groups = num_vars
    else:
        groups, unique_group_names = compute_groups_matrix(groups)
        number_of_groups = len(set(unique_group_names))
    # End if

    num_trajectories = int(Y.size / (number_of_groups + 1))
    trajectory_size = int(Y.size / num_trajectories)

    elementary_effects, see_trajectory = _compute_elementary_effects(X, SX, Y, trajectory_size, delta)

    Si = _compute_statistical_outputs(
        elementary_effects,
        num_vars,
        num_resamples,
        conf_level,
        groups,
        unique_group_names, 
        see_trajectory,
    )

    if print_to_console:
        print(Si.to_df())

    return Si


def _compute_statistical_outputs(
    elementary_effects: np.ndarray,
    num_vars: int,
    num_resamples: int,
    conf_level: float,
    groups: np.ndarray,
    unique_group_names: List,
    see_trajectory: np.ndarray
) -> ResultDict:
    """Computes the statistical parameters related to Morris method.

    Parameters
    ----------
    elementary_effects: np.ndarray
        Morris elementary effects.
    num_vars: int
        Number of problem's variables
    num_resamples: int
        Number of resamples
    conf_level: float
        Confidence level
    groups: np.ndarray
        Array defining the distribution of groups
    unique_group_names: List
        Names of the groups

    Returns
    -------
    Si: ResultDict
        Morris statistical parameters.
    """

    Si = ResultDict(
        (k, [None] * num_vars)
        for k in ["names", "mu", "mu_star", "sigma", "mu_star_conf", "standardised_EE"]
    )

    mu = np.average(elementary_effects, 1)
    mu_star = np.average(np.abs(elementary_effects), 1)
    sigma = np.std(elementary_effects, axis=1, ddof=1)
    mu_star_conf = _compute_mu_star_confidence(
        elementary_effects, num_vars, num_resamples, conf_level)
    standardised_EE = np.average(np.abs(see_trajectory), 1)

    Si["names"] = unique_group_names
    Si["mu"] = _compute_grouped_sigma(mu, groups)
    Si["mu_star"] = _compute_grouped_metric(mu_star, groups)
    Si["sigma"] = _compute_grouped_sigma(sigma, groups)
    Si["mu_star_conf"] = _compute_grouped_metric(mu_star_conf, groups)
    Si["standardised_EE"] = standardised_EE

    return Si


def _compute_grouped_sigma(
    ungrouped_sigma: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    """Sigma values for the groups.

    Returns sigma for the groups of parameter values in the argument
    ungrouped_metric where the group consists of no more than
    one parameter

    Parameters
    ----------
    ungrouped_sigma: np.ndarray
        Sigma values calculated without considering the groups
    groups: np.ndarray
        Array defining the distribution of groups

    Returns
    -------
    sigma: np.ndarray
        Sigma values for the groups.
    """
    sigma_agg = _compute_grouped_metric(ungrouped_sigma, groups)

    sigma = np.zeros(groups.shape[1], dtype=float)
    np.copyto(sigma, sigma_agg, where=groups.sum(axis=0) == 1)
    np.copyto(sigma, np.NAN, where=groups.sum(axis=0) != 1)

    return sigma


def _compute_grouped_metric(
    ungrouped_metric: np.ndarray, groups: np.ndarray
) -> np.ndarray:
    """Computes the mean value for the groups of parameter values.

    Parameters
    ----------
    ungrouped_metric: np.ndarray
        Metric calculated without considering the groups
    groups: np.ndarray
        Array defining the distribution of groups

    Returns
    -------
    mean_of_mu_star: np.ndarray
         Mean value for the groups of parameter values
    """

    groups = np.array(groups, dtype=bool)

    mu_star_masked = np.ma.masked_array(
        ungrouped_metric * groups.T, mask=(groups ^ 1).T
    )
    mean_of_mu_star = np.ma.mean(mu_star_masked, axis=1)

    return mean_of_mu_star


def _reorganize_output_matrix(
    output_array: np.ndarray,
    value_increased: np.ndarray,
    value_decreased: np.ndarray,
    increase: bool = True,
) -> np.ndarray:
    """Reorganize the output matrix.

    This method reorganizes the output matrix in a way that allows the
    elementary effects to be computed as a simple subtraction between two
    arrays. It repositions the outputs in the output matrix according to the
    order they changed during the formation of the trajectories.

    Parameters
    ----------
    output_array: np.ndarray
        Matrix of model output values
    value_increased: np.ndarray
        Input variables that had their values increased when forming the
        trajectories matrix
    value_decreased: np.ndarray
        Input variables that had their values decreased when forming the
        trajectories matrix
    increase: bool
        Direction to consider (values that increased or decreased). "Increase"
        is the default value.
    Returns
    -------

    """
    if increase:
        pad_up = (1, 0)
        pad_lo = (0, 1)
    else:
        pad_up = (0, 1)
        pad_lo = (1, 0)

    value_increased = np.pad(value_increased, ((0, 0), pad_up, (0, 0)), "constant")
    value_decreased = np.pad(value_decreased, ((0, 0), pad_lo, (0, 0)), "constant")

    res = np.einsum("ik,ikj->ij", output_array, value_increased + value_decreased)

    return res.T


def _compute_elementary_effects(
    model_inputs: np.ndarray,
    nominal_model_inputs: np.ndarray,
    model_outputs: np.ndarray,
    trajectory_size: int,
    delta: float,
) -> np.ndarray:
    """Computes the Morris elementary effects.

    Parameters
    ----------
    model_inputs: np.ndarray
        matrix of inputs to the model under analysis.
        x-by-r where x is the number of variables and
        r is the number of rows (a function of x and num_trajectories)
    nominal_model_inputs: np.ndarray
        matrix of nominal inputs to the model under analysis.
        x-by-r where x is the number of variables and
        r is the number of rows (a function of x and num_trajectories)
    model_outputs: np.ndarray
        r-length vector of model outputs
    trajectory_size: int
        Number of points in a trajectory
    delta: float
        Scaling factor computed from `num_levels`

    Returns
    ---------
    elementary_effects : np.array
        Elementary effects for each parameter
    """
    num_trajectories = _calculate_number_trajectories(model_inputs, trajectory_size)

    output_matrix = _reshape_model_outputs(
        model_outputs, num_trajectories, trajectory_size
    )
    input_matrix = _reshape_model_inputs(
        model_inputs, num_trajectories, trajectory_size
    )

    delta_variables = _calculate_delta_input_variables(input_matrix)
    value_increased = delta_variables > 0
    value_decreased = delta_variables < 0

    result_increased = _reorganize_output_matrix(
        output_matrix, value_increased, value_decreased
    )
    result_decreased = _reorganize_output_matrix(
        output_matrix, value_increased, value_decreased, increase=False
    )

    elementary_effects = _calc_results_difference(result_increased, result_decreased)
    np.divide(elementary_effects, delta, out=elementary_effects)
    
    def _calculate_step_size_x(input_matrix):
        """
        This function calculates the step of the dx at nominal values
        """
        max_y = np.max(input_matrix, axis=1)
        min_y = np.min(input_matrix, axis=1)

        delta = max_y - min_y
        return delta
    
    see_dy = _calc_results_difference(result_increased, result_decreased)

    input_matrix_nominal = _reshape_model_inputs(
    nominal_model_inputs, num_trajectories, trajectory_size
    )

    dx_trajectory = _calculate_step_size_x(input_matrix_nominal)
    
    see_dy_dx = np.divide(see_dy, dx_trajectory.T)

    sigma_x_pertrajectory = np.std(input_matrix_nominal, axis=1)
    sigma_y_pertrajectory = np.std(output_matrix, axis=1)

    divisor = sigma_y_pertrajectory

    adjustment_trajectory = sigma_x_pertrajectory / divisor[:, np.newaxis]

    scaled_elementary_effect = see_dy_dx*adjustment_trajectory.T

    return elementary_effects, scaled_elementary_effect

def _calc_results_difference(
    result_up: np.ndarray, result_lo: np.ndarray
) -> np.ndarray:
    """Computes the difference between the output points.

    Parameters
    ----------
    result_up: np.ndarray
    result_lo: np.ndarray

    Returns
    -------
    results_difference: np.ndarray
        Difference between the output points
    """
    results_difference = np.subtract(result_up, result_lo)

    return results_difference


def _calculate_delta_input_variables(input_matrix: np.ndarray) -> np.ndarray:
    """Computes the delta values of the problem variables.

    For each point of the trajectory, computes how much each variable increased
    or decreased in respect to the previous point.

    Parameters
    ----------
    input_matrix: np.ndarray
        Matrix with the values of the problem's variables for all input points.

    Returns
    -------
    delta_variables: np.ndarray
        Variation of each variable, for each point in the trajectory.
    """
    delta_variables = np.subtract(input_matrix[:, 1:, :], input_matrix[:, 0:-1, :])

    return delta_variables


def _calculate_number_trajectories(
    model_inputs: np.ndarray, trajectory_size: int
) -> int:
    """Calculate the number of trajectories.

    Parameters
    ----------
    model_inputs: np.ndarray
        Matrix of model inputs
    trajectory_size: int
        Number of input points in each trajectory

    Returns
    -------
    num_trajectories: int
        Number of trajectories
    """
    num_input_points = model_inputs.shape[0]
    num_trajectories = int(num_input_points / trajectory_size)

    return num_trajectories


def _reshape_model_inputs(
    model_inputs: np.ndarray, num_trajectories: int, trajectory_size: int
) -> np.ndarray:
    """Reshapes the model inputs' matrix.

    Parameters
    ----------
    model_inputs: np.ndarray
        Matrix of model inputs
    num_trajectories: int
        Number of trajectories
    trajectory_size: int
        Number of points in a trajectory

    Returns
    -------
    input_matrix: np.ndarray
        Reshaped input matrix.
    """
    num_vars = model_inputs.shape[1]
    input_matrix = model_inputs.reshape(num_trajectories, trajectory_size, num_vars)
    return input_matrix


def _reshape_model_outputs(
    model_outputs: np.ndarray, num_trajectories: int, trajectory_size: int
):
    """Reshapes the model outputs' matrix.

    Parameters
    ----------
    model_outputs: np.ndarray
        Matrix of model outputs
    num_trajectories: int
        Number of trajectories
    trajectory_size: int
        Number of points in a trajectory

    Returns
    -------
    output_matrix: np.ndarray
        Reshaped output matrix.
    """
    output_matrix = model_outputs.reshape(num_trajectories, trajectory_size)
    return output_matrix


def _compute_mu_star_confidence(
    elementary_effects: np.ndarray, num_vars: int, num_resamples: int, conf_level: float
) -> np.ndarray:
    """Computes the confidence intervals for the mu_star variable.

    Uses bootstrapping where the elementary effects are resampled with
    replacement to produce a histogram of resampled mu_star metrics.
    This resample is used to produce a confidence interval.

    Parameters
    ----------
    elementary_effects : np.array
        Elementary effects for each parameter
    num_vars: int
        Number of problem's variables
    num_resamples: int
        Number of resamples
    conf_level: float
        Confidence level

    Returns
    -------
    mu_star_conf: np.ndarray
        Confidence intervals for the mu_star variable
    """
    if not 0 < conf_level < 1:
        raise ValueError("Confidence level must be between 0-1.")

    mu_star_conf = []
    for j in range(num_vars):
        ee = elementary_effects[j, :]
        resample_index = np.random.randint(len(ee), size=(num_resamples, len(ee)))
        ee_resampled = ee[resample_index]

        # Compute average of the absolute values over each of the resamples
        mu_star_resampled = np.average(np.abs(ee_resampled), axis=1)

        mu_star_conf.append(
            norm.ppf(0.5 + conf_level / 2) * mu_star_resampled.std(ddof=1)
        )

    mu_star_conf = np.asarray(mu_star_conf)

    return mu_star_conf


def _check_if_array_of_floats(array_x: np.ndarray):
    """Checks if an arrays is made of floats. If not, raises an error.

    Parameters
    ----------
    array_x:
        Array to be checked
    """
    msg = "dtype of {} array must be 'float', float32 or float64"
    if array_x.dtype not in ["float", "float32", "float64"]:
        raise ValueError(msg.format(array_x))


def cli_parse(parser):
    parser.add_argument(
        "-X",
        "--model-input-file",
        type=str,
        required=True,
        default=None,
        help="Model input file",
    )
    parser.add_argument(
        "-r",
        "--resamples",
        type=int,
        required=False,
        default=1000,
        help="Number of bootstrap resamples for Sobol \
                           confidence intervals",
    )
    parser.add_argument(
        "-l",
        "--levels",
        type=int,
        required=False,
        default=4,
        help="Number of grid levels \
                           (Morris only)",
    )
    parser.add_argument(
        "--grid-jump",
        type=int,
        required=False,
        default=2,
        help="Grid jump size (Morris only)",
    )
    return parser


def cli_action(args):
    problem = read_param_file(args.paramfile)
    Y = np.loadtxt(
        args.model_output_file, delimiter=args.delimiter, usecols=(args.column,)
    )
    X = np.loadtxt(args.model_input_file, delimiter=args.delimiter, ndmin=2)
    if len(X.shape) == 1:
        X = X.reshape((len(X), 1))
    NOMX = np.loadtxt(args.model_input_file, delimiter=args.delimiter, ndmin=2)
    if len(NOMX.shape) == 1:
        NOMX = NOMX.reshape((len(NOMX), 1))

    analyze(
        problem,
        X,
        NOMX,
        Y,
        num_resamples=args.resamples,
        print_to_console=True,
        num_levels=args.levels,
        seed=args.seed,
    )


if __name__ == "__main__":
    common_args.run_cli(cli_parse, cli_action)
