from __future__ import division
from __future__ import print_function
from ..util import read_param_file
from sys import exit
import numpy as np
from scipy.stats import norm
from . import common_args

# Perform Morris Analysis on file of model results
# Returns a dictionary with keys 'mu', 'mu_star', 'sigma', and 'mu_star_conf'
# Where each entry is a list of size D (the number of parameters)
# Containing the indices in the same order as the parameter file


def analyze(pfile, input_file, output_file, column=0, delim=' ', num_resamples=1000, conf_level=0.95, print_to_console=False):

    param_file = read_param_file(pfile)
    Y = np.loadtxt(output_file, delimiter=delim, usecols=(column,))
    X = np.loadtxt(input_file, delimiter=delim, ndmin=2)
    if len(X.shape) == 1:
        X = X.reshape((len(X), 1))

    D = param_file['num_vars']

    if Y.size % (D + 1) == 0:
        N = int(Y.size / (D + 1))
    else:
        print("""
                Error: Number of samples in model output file must be a multiple of (D+1), 
                where D is the number of parameters in your parameter file.
              """)
        exit()

    ee = np.empty([N, D])

    # For each of the N trajectories
    for i in range(N):

        # Set up the indices corresponding to this trajectory
        j = np.arange(D + 1) + i * (D + 1)
        j1 = j[0:D]
        j2 = j[1:D + 1]

        # The elementary effect is (change in output)/(change in input)
        # Each parameter has one EE per trajectory, because it is only changed
        # once in each trajectory
        ee[i, :] = np.linalg.solve(X[j2,:]-X[j1,:], Y[j2]-Y[j1])

    # Rescale elementary effects so that the delta step is in the quantile
    # space [0,1]
    for j in range(D):
        lb = param_file['bounds'][j][0]
        ub = param_file['bounds'][j][1]
        ee[:, j] *= (ub - lb)

    # Output the Mu, Mu*, and Sigma Values. Also return them in case this is
    # being called from Python
    Si = dict((k, [None] * D)
              for k in ['mu', 'mu_star', 'sigma', 'mu_star_conf'])
    if print_to_console:
        print("Parameter Mu Sigma Mu_Star Mu_Star_Conf")

    for j in range(D):
        Si['mu'][j] = np.average(ee[:, j])
        Si['mu_star'][j] = np.average(np.abs(ee[:, j]))
        Si['sigma'][j] = np.std(ee[:, j])
        Si['mu_star_conf'][j] = compute_mu_star_confidence(
            ee[:, j], N, num_resamples, conf_level)

        if print_to_console:
            print("%s %f %f %f %f" % (param_file['names'][j], Si['mu'][j], Si[
                  'sigma'][j], Si['mu_star'][j], Si['mu_star_conf'][j]))

    return Si


def compute_mu_star_confidence(ee, N, num_resamples, conf_level):

    ee_resampled = np.empty([N])
    mu_star_resampled = np.empty([num_resamples])

    if conf_level < 0 or conf_level > 1:
        print("Error: Confidence level must be between 0-1.")
        exit()

    for i in range(num_resamples):
        for j in range(N):

            index = np.random.randint(0, N)
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
