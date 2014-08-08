from __future__ import division
import numpy as np
from scipy.stats import norm
from ..util import read_param_file
import common_args

# Perform Sobol Analysis on file of model results
# Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf'
# Where each entry is a list of size D (the number of parameters)
# Containing the indices in the same order as the parameter file
def analyze(pfile, output_file, column = 0, calc_second_order = True, num_resamples = 1000,
            delim = ' ', conf_level = 0.95, print_to_console=False):
    
    param_file = read_param_file(pfile)
    Y = np.loadtxt(output_file, delimiter=delim, usecols=(column,))
    D = param_file['num_vars']

    if calc_second_order and Y.size % (2*D + 2) == 0:
        N = int(Y.size / (2*D + 2))
    elif not calc_second_order and Y.size % (D + 2) == 0:
        N = int(Y.size / (D + 2))
    else: raise RuntimeError("""
        Incorrect number of samples in model output file. 
        Confirm that calc_second_order matches option used during sampling.""")  

    if conf_level < 0 or conf_level > 1: raise RuntimeError("Confidence level must be between 0-1.")
    
    A = np.empty(N)
    B = np.empty(N)
    AB = np.empty((N,D))
    BA = np.empty((N,D)) if calc_second_order else None
    step = 2*D+2 if calc_second_order else D+2

    A = Y[0:Y.size:step]
    B =  Y[(step-1):Y.size:step]
    for j in xrange(D):
        AB[:,j] = Y[(j+1):Y.size:step]
        if calc_second_order: BA[:,j] = Y[(j+1+D):Y.size:step]
    
    # First order (+conf.) and Total order (+conf.)
    keys = ('S1','S1_conf','ST','ST_conf')
    S = dict((k, np.empty(D)) for k in keys)
    if print_to_console: print "Parameter %s %s %s %s" % keys

    for j in xrange(D):
        S['S1'][j] = first_order(A, AB[:,j], B)
        S['S1_conf'][j] = first_order_confidence(A, AB[:,j], B, num_resamples, conf_level)
        S['ST'][j] = total_order(A, AB[:,j], B)
        S['ST_conf'][j] = total_order_confidence(A, AB[:,j], B, num_resamples, conf_level)
        
        if print_to_console:
            print "%s %f %f %f %f" % (param_file['names'][j], S['S1'][j], S['S1_conf'][j], S['ST'][j], S['ST_conf'][j])
    
    # Second order (+conf.)
    if calc_second_order:
        S['S2'] = np.empty((D,D))
        S['S2_conf'] = np.empty((D,D))
        if print_to_console: print "\nParameter_1 Parameter_2 S2 S2_conf"
        
        for j in range(D):
            for k in range(j+1, D):     
                S['S2'] = second_order(A, BA[:,j], AB[:,k], AB[:,j], B, N)
                S['S2_conf'] = second_order_confidence(A, BA[:,j], AB[:,k], AB[:,j], B, N, num_resamples, conf_level)
                
                if print_to_console:
                    print "%s %s %f %f" % (param_file['names'][j], param_file['names'][k], S['S2'], S['S2_conf'])                        
    
    return S            
        
def first_order(A, AB, B):
    # First order estimator following Saltelli et al. 2010 CPC, normalized by sample variance
    return np.mean(B*(AB-A))/np.var(np.r_[A,B])

def first_order_confidence(A, AB, B, num_resamples, conf_level):
    s  = np.empty(num_resamples)
    for i in xrange(num_resamples):        
        r = np.random.randint(len(A), size=len(A))        
        s[i] = first_order(A[r], AB[r], B[r])
    
    return norm.ppf(0.5 + conf_level/2) * s.std(ddof=1)

def total_order(A, AB, B):
    # Total order estimator following Saltelli et al. 2010 CPC, normalized by sample variance
    return 0.5*np.mean((A-AB)**2)/np.var(np.r_[A,B])

def total_order_confidence(A, AB, B, num_resamples, conf_level):
    s  = np.empty(num_resamples)    
    for i in xrange(num_resamples):
        r = np.random.randint(len(A), size=len(A))  
        s[i] = total_order(A[r], AB[r], B[r])
    
    return norm.ppf(0.5 + conf_level/2) * s.std(ddof=1)

def second_order(a0, a1, a2, a3, a4, N):
    
    c = np.average(a0)    
    EY = np.mean((a0-c)*(a4-c))
    EY2 = np.mean((a1-c)*(a3-c))
    
    V = np.sum((a1-c)**2)/(N-1) - np.mean(a1-c)**2
    Vij = np.sum((a1-c)*(a2-c))/(N-1) - EY2
    Vi = np.sum((a2-c)*(a4-c))/(N-1) - EY
    Vj = np.sum((a3-c)*(a4-c))/(N-1) - EY2
    
    return (Vij - Vi - Vj) / V

def second_order_confidence(a0, a1, a2, a3, a4, N, num_resamples, conf_level):
    
    s  = np.empty([num_resamples])
    
    for i in range(num_resamples):
        r = np.random.randint(N, size=N)
        s[i] = second_order(a0[r], a1[r], a2[r], a3[r], a4[r], N)
    
    return norm.ppf(0.5 + conf_level/2) * s.std(ddof=1)

if __name__ == "__main__":

    parser = common_args.create()
    parser.add_argument('--max-order', type=int, required=False, default=2, choices=[1, 2], help='Maximum order of sensitivity indices to calculate')
    parser.add_argument('-r', '--resamples', type=int, required=False, default=1000, help='Number of bootstrap resamples for Sobol confidence intervals')
    args = parser.parse_args()

    analyze(args.paramfile, args.model_output_file, args.column, (args.max_order == 2), num_resamples = args.resamples, delim = args.delimiter, print_to_console=True)
