from __future__ import division
from __future__ import print_function
import numpy as np
from scipy.stats import norm
from ..util import read_param_file
from . import common_args

# Calculate DGSM from file of model results
# Returns a dictionary with keys 'vi', 'vi_std', 'dgsm', and 'dgsm_conf'
# Where each entry is a list of size D (the number of parameters)
# Containing the indices in the same order as the parameter file
def analyze(pfile, input_file, output_file, column = 0, num_resamples = 1000,
            delim = ' ', conf_level = 0.95, print_to_console=False):
    
    pf = read_param_file(pfile)
    Y = np.loadtxt(output_file, delimiter=delim, usecols=(column,))
    D = pf['num_vars']
    X = np.loadtxt(input_file, delimiter=delim, ndmin=2)
    if len(X.shape) == 1: X = X.reshape((len(X),1))

    if Y.size % (D + 1) == 0:
        N = int(Y.size / (D + 1))
    else: raise RuntimeError("Incorrect number of samples in model output file.")  

    if conf_level < 0 or conf_level > 1: raise RuntimeError("Confidence level must be between 0-1.")

    base = np.empty(N)
    X_base = np.empty((N,D))
    perturbed = np.empty((N,D))
    X_perturbed = np.empty((N,D))
    step = D+1

    base = Y[0:Y.size:step]
    X_base = X[0:Y.size:step,:]
    for j in range(D):
        perturbed[:,j] = Y[(j+1):Y.size:step]
        X_perturbed[:,j] = X[(j+1):Y.size:step,j]

    # First order (+conf.) and Total order (+conf.)
    keys = ('vi', 'vi_std', 'dgsm', 'dgsm_conf')
    S = dict((k, np.empty(D)) for k in keys)
    if print_to_console: print("Parameter %s %s %s %s" % keys)

    for j in range(D):
        S['vi'][j], S['vi_std'][j] = calc_vi(base, perturbed[:,j], X_perturbed[:,j]-X_base[:,j])
        S['dgsm'][j], S['dgsm_conf'][j] = calc_dgsm(base, perturbed[:,j], X_perturbed[:,j]-X_base[:,j], pf['bounds'][j], num_resamples, conf_level)

        if print_to_console:
            print("%s %f %f %f %f" % (pf['names'][j], S['vi'][j], S['vi_std'][j], S['dgsm'][j], S['dgsm_conf'][j]))

    return S

def calc_vi(base, perturbed, x_delta):
    # v_i sensitivity measure following Sobol and Kucherenko (2009)
    # For comparison, Morris mu* < sqrt(v_i)
    dfdx = (perturbed-base)/x_delta
    dfdx2 = dfdx**2

    return np.mean(dfdx2), np.std(dfdx2)

def calc_dgsm(base, perturbed, x_delta, bounds, num_resamples, conf_level):
    # v_i sensitivity measure following Sobol and Kucherenko (2009)
    # For comparison, total order S_tot <= dgsm
    D = np.var(base)
    vi, _ = calc_vi(base, perturbed, x_delta)
    dgsm = vi*(bounds[1]-bounds[0])**2/(D*np.pi**2)

    s = np.empty(num_resamples)
    for i in range(num_resamples):        
        r = np.random.randint(len(base), size=len(base))        
        s[i], _ = calc_vi(base[r], perturbed[r], x_delta[r])

    return dgsm, norm.ppf(0.5 + conf_level/2) * s.std(ddof=1)

if __name__ == "__main__":
    parser = common_args.create()
    parser.add_argument('-X', '--model-input-file', type=str, required=True, default=None, help='Model input file')
    parser.add_argument('-r', '--resamples', type=int, required=False, default=1000, help='Number of bootstrap resamples for Sobol confidence intervals')
    args = parser.parse_args()

    analyze(args.paramfile, args.model_input_file, args.model_output_file, args.column, 
        num_resamples = args.resamples, delim = args.delimiter, print_to_console=True)
