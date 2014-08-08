from __future__ import division
import numpy as np
<<<<<<< HEAD
import math
from scipy.stats import norm
=======
from scipy.stats import norm
from ..util import read_param_file
import common_args
>>>>>>> 1faf68b7a8c74f7b3ed79a1b17414c64943cb6a9

# Perform Sobol Analysis on file of model results
# Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf'
# Where each entry is a list of size D (the number of parameters)
# Containing the indices in the same order as the parameter file
<<<<<<< HEAD
def analyze(pfile, output_file, column = 0, calc_second_order = True, num_resamples = 1000, delim = ' ', conf_level = 0.95):
=======
def analyze(pfile, output_file, column = 0, calc_second_order = True, num_resamples = 1000,
            delim = ' ', conf_level = 0.95, print_to_console=False):
>>>>>>> 1faf68b7a8c74f7b3ed79a1b17414c64943cb6a9
    
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
<<<<<<< HEAD
    Si = dict((k, [None]*D) for k in ['S1','S1_conf','ST','ST_conf'])
    print "Parameter First_Order First_Order_Conf Total_Order Total_Order_Conf"
    for j in range(D):
        a0 = np.empty([N])
        a1 = np.empty([N])
        a2 = np.empty([N])
        
        for i in range(N):
            a0[i] = A[i]
            a1[i] = C_A[i][j]
            a2[i] = B[i]
            
        Si['S1'][j] = compute_first_order(a0, a1, a2, N)
        Si['S1_conf'][j] = compute_first_order_confidence(a0, a1, a2, N, num_resamples, conf_level)
        Si['ST'][j] = compute_total_order(a0, a1, a2, N)
        Si['ST_conf'][j] = compute_total_order_confidence(a0, a1, a2, N, num_resamples, conf_level)
        
        print "%s %f %f %f %f" % (param_file['names'][j], Si['S1'][j], Si['S1_conf'][j], Si['ST'][j], Si['ST_conf'][j])
=======
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
>>>>>>> 1faf68b7a8c74f7b3ed79a1b17414c64943cb6a9
    
    # Second order (+conf.)
    if calc_second_order:
        S['S2'] = np.empty((D,D)); S['S2'][:] = np.nan
        S['S2_conf'] = np.empty((D,D)); S['S2_conf'][:] = np.nan
        if print_to_console: print "\nParameter_1 Parameter_2 S2 S2_conf"
        
        for j in range(D):
<<<<<<< HEAD
            for k in range(j+1, D):
                a0 = np.empty([N])
                a1 = np.empty([N])
                a2 = np.empty([N])
                a3 = np.empty([N])
                a4 = np.empty([N])
                
                for i in range(N):
                    a0[i] = A[i]
                    a1[i] = C_B[i][j]
                    a2[i] = C_A[i][k]
                    a3[i] = C_A[i][j]
                    a4[i] = B[i]
                    
                S2 = compute_second_order(a0, a1, a2, a3, a4, N)
                S2c = compute_second_order_confidence(a0, a1, a2, a3, a4, N, num_resamples, conf_level)
                
                print "%s %s %f %f" % (param_file['names'][j], param_file['names'][k], S2, S2c)                        
    
    return Si            
=======
            for k in range(j+1, D):     
                S['S2'][j,k] = second_order(A, AB[:,j], AB[:,k], BA[:,j], B)
                S['S2_conf'][j,k] = second_order_confidence(A, AB[:,j], AB[:,k], BA[:,j], B, num_resamples, conf_level)
                
                if print_to_console:
                    print "%s %s %f %f" % (param_file['names'][j], param_file['names'][k], S['S2'][j,k], S['S2_conf'][j,k])                        
    
    return S            
>>>>>>> 1faf68b7a8c74f7b3ed79a1b17414c64943cb6a9
        
def first_order(A, AB, B):
    # First order estimator following Saltelli et al. 2010 CPC, normalized by sample variance
    return np.mean(B*(AB-A))/np.var(np.r_[A,B])

def first_order_confidence(A, AB, B, num_resamples, conf_level):
    s  = np.empty(num_resamples)
    for i in xrange(num_resamples):        
        r = np.random.randint(len(A), size=len(A))        
        s[i] = first_order(A[r], AB[r], B[r])
    
    return norm.ppf(0.5 + conf_level/2) * s.std(ddof=1)

<<<<<<< HEAD
def compute_first_order_confidence(a0, a1, a2, N, num_resamples, conf_level):
    
    b0 = np.empty([N])
    b1 = np.empty([N])
    b2 = np.empty([N])
    s  = np.empty([num_resamples])

    if conf_level < 0 or conf_level > 1:    
        print "Error: Confidence level must be between 0-1."
        exit() 
    
    for i in range(num_resamples):
        for j in range(N):
            
            index = np.random.randint(0, N)
            b0[j] = a0[index]
            b1[j] = a1[index]
            b2[j] = a2[index]
        
        s[i] = compute_first_order(b0, b1, b2, N)
    
    return norm.ppf(0.5 + conf_level/2) * s.std(ddof=1)
=======
def total_order(A, AB, B):
    # Total order estimator following Saltelli et al. 2010 CPC, normalized by sample variance
    return 0.5*np.mean((A-AB)**2)/np.var(np.r_[A,B])
>>>>>>> 1faf68b7a8c74f7b3ed79a1b17414c64943cb6a9

def total_order_confidence(A, AB, B, num_resamples, conf_level):
    s  = np.empty(num_resamples)    
    for i in xrange(num_resamples):
        r = np.random.randint(len(A), size=len(A))  
        s[i] = total_order(A[r], AB[r], B[r])
    
    return norm.ppf(0.5 + conf_level/2) * s.std(ddof=1)

<<<<<<< HEAD
def compute_total_order_confidence(a0, a1, a2, N, num_resamples, conf_level):
    
    b0 = np.empty([N])
    b1 = np.empty([N])
    b2 = np.empty([N])
    s  = np.empty([num_resamples])

    if conf_level < 0 or conf_level > 1:    
        print "Error: Confidence level must be between 0-1."
        exit() 
    
    for i in range(num_resamples):
        for j in range(N):
            
            index = np.random.randint(0, N)
            b0[j] = a0[index]
            b1[j] = a1[index]
            b2[j] = a2[index]
        
        s[i] = compute_total_order(b0, b1, b2, N)
    
    return norm.ppf(0.5 + conf_level/2) * s.std(ddof=1)
=======
def second_order(A, ABj, ABk, BAj, B):
    # Second order estimator following Saltelli 2002
    V = np.var(np.r_[A,B])
    Vjk = np.mean(BAj*ABk - A*B)
    Sj = first_order(A,ABj,B)
    Sk = first_order(A,ABk,B)
    
    return Vjk/V - Sj - Sk
>>>>>>> 1faf68b7a8c74f7b3ed79a1b17414c64943cb6a9

def second_order_confidence(A, ABj, ABk, BAj, B, num_resamples, conf_level):
    s  = np.empty(num_resamples)
    for i in xrange(num_resamples):
        r = np.random.randint(len(A), size=len(A))
        s[i] = second_order(A[r], ABj[r], ABk[r], BAj[r], B[r])
    
    return norm.ppf(0.5 + conf_level/2) * s.std(ddof=1)

if __name__ == "__main__":
    parser = common_args.create()
    parser.add_argument('--max-order', type=int, required=False, default=2, choices=[1, 2], help='Maximum order of sensitivity indices to calculate')
    parser.add_argument('-r', '--resamples', type=int, required=False, default=1000, help='Number of bootstrap resamples for Sobol confidence intervals')
    args = parser.parse_args()

<<<<<<< HEAD
def compute_second_order_confidence(a0, a1, a2, a3, a4, N, num_resamples, conf_level):
    
    b0 = np.empty([N])
    b1 = np.empty([N])
    b2 = np.empty([N])
    b3 = np.empty([N])
    b4 = np.empty([N])
    s  = np.empty([num_resamples])

    if conf_level < 0 or conf_level > 1:    
        print "Error: Confidence level must be between 0-1."
        exit() 
    
    for i in range(num_resamples):
        for j in range(N):
            
            index = np.random.randint(0, N)
            b0[j] = a0[index]
            b1[j] = a1[index]
            b2[j] = a2[index]
            b3[j] = a3[index]
            b4[j] = a4[index]
        
        s[i] = compute_second_order(b0, b1, b2, b3, b4, N)
    
    return norm.ppf(0.5 + conf_level/2) * s.std(ddof=1)
=======
    analyze(args.paramfile, args.model_output_file, args.column, (args.max_order == 2), 
        num_resamples = args.resamples, delim = args.delimiter, print_to_console=True)
>>>>>>> 1faf68b7a8c74f7b3ed79a1b17414c64943cb6a9
