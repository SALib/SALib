from __future__ import division
import numpy as np
from scipy.stats import norm, gaussian_kde, rankdata
from ..util import read_param_file
import common_args

# Perform Sobol Analysis on file of model results
# Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf'
# Where each entry is a list of size D (the number of parameters)
# Containing the indices in the same order as the parameter file
def analyze(pfile, input_file, output_file, column = 0, calc_second_order = True, num_resamples = 10,
            delim = ' ', conf_level = 0.95, print_to_console=False):
    
    param_file = read_param_file(pfile)
    Y = np.loadtxt(output_file, delimiter=delim, usecols=(column,))
    X = np.loadtxt(input_file, delimiter=delim, ndmin=2)
    if len(X.shape) == 1: X = X.reshape((len(X),1))

    D = param_file['num_vars']
    N = Y.size

    if conf_level < 0 or conf_level > 1: raise RuntimeError("Confidence level must be between 0-1.")

    # equal frequency partition
    M = min(np.ceil(N**(2/(7+np.tanh((1500-N)/500)))), 48);
    m = np.linspace(0,N,M+1);
    Ygrid = np.linspace(np.min(Y), np.max(Y), 100)

    keys = ('delta', 'delta_conf', 'S1', 'S1_conf')
    S = dict((k, np.zeros(D)) for k in keys)
    if print_to_console: print "Parameter %s %s %s %s" % keys
    
    for i in xrange(D):
        S['delta'][i], S['delta_conf'][i] = bias_reduced_delta(Y, Ygrid, X[:,i], m, num_resamples, conf_level)
        S['S1'][i] = sobol_first(Y, X[:,i], m)
        S['S1_conf'][i] = sobol_first_conf(Y, X[:,i], m, num_resamples, conf_level)
        if print_to_console:
            print "%s %f %f %f %f" % (param_file['names'][i], S['delta'][i], S['delta_conf'][i], S['S1'][i], S['S1_conf'][i])
    
    return S       

# Plischke et al. 2013 estimator (eqn 26) for d_hat
def calc_delta(Y, Ygrid, X, m):
    N = len(Y);
    fy = gaussian_kde(Y, bw_method='silverman')(Ygrid)
    xr = rankdata(X, method='ordinal')
    
    d_hat = 0
    for j in xrange(len(m)-1):
        ix = np.where((xr > m[j]) & (xr <= m[j+1]))[0]
        nm = len(ix)
        fyc = gaussian_kde(Y[ix], bw_method='silverman')(Ygrid)
        d_hat += (nm/(2*N))*np.trapz(np.abs(fy-fyc), Ygrid)

    return d_hat

# Plischke et al. 2013 bias reduction technique (eqn 30)
def bias_reduced_delta(Y, Ygrid, X, m, num_resamples, conf_level):
    d  = np.empty(num_resamples)
    d_hat = calc_delta(Y, Ygrid, X, m)

    for i in xrange(num_resamples):        
        r = np.random.randint(len(Y), size=len(Y))        
        d[i] = calc_delta(Y[r], Ygrid, X[r], m)

    d = 2*d_hat - d
    return (d.mean(), norm.ppf(0.5 + conf_level/2) * d.std(ddof=1)) 

def sobol_first(Y, X, m):
    xr = rankdata(X, method='ordinal')
    Vi = 0
    N = len(Y)
    for j in xrange(len(m)-1):
        ix = np.where((xr > m[j]) & (xr <= m[j+1]))[0]
        nm = len(ix)
        Vi += (nm/N)*(Y[ix].mean() - Y.mean())**2
    return Vi/np.var(Y)

def sobol_first_conf(Y, X, m, num_resamples, conf_level):
    s = np.empty(num_resamples)

    for i in xrange(num_resamples):
        r = np.random.randint(len(Y), size=len(Y))
        s[i] = sobol_first(Y[r], X[r], m)

    return norm.ppf(0.5 + conf_level/2) * s.std(ddof=1)

if __name__ == "__main__":
    parser = common_args.create()
    parser.add_argument('-X', '--model-input-file', type=str, required=True, default=None, help='Model input file')
    parser.add_argument('-r', '--resamples', type=int, required=False, default=10, help='Number of bootstrap resamples for Sobol confidence intervals')
    args = parser.parse_args()

    args = parser.parse_args()
    analyze(args.paramfile, args.model_input_file, args.model_output_file, args.column, 
        delim=args.delimiter, num_resamples = args.resamples, print_to_console=True)
