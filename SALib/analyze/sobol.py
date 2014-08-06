from __future__ import division
from ..util import read_param_file
from sys import exit
import numpy as np
from scipy.stats import norm
import common_args

# Perform Sobol Analysis on file of model results
# Returns a dictionary with keys 'S1', 'S1_conf', 'ST', and 'ST_conf'
# Where each entry is a list of size D (the number of parameters)
# Containing the indices in the same order as the parameter file
def analyze(pfile, output_file, column = 0, calc_second_order = True, num_resamples = 1000, delim = ' ', conf_level = 0.95, print_to_console=False):
    
    param_file = read_param_file(pfile)
    Y = np.loadtxt(output_file, delimiter=delim, usecols=(column,))
    D = param_file['num_vars']

    if calc_second_order:
        if Y.size % (2*D + 2) == 0:
            N = int(Y.size / (2*D + 2))
        else:
            print """
                Error: Number of samples in model output file must be a multiple of (2D+2), 
                where D is the number of parameters in your parameter file.
                (You have calc_second_order set to true, which is true by default.)
              """
            exit()
    else:
        if Y.size % (D + 2) == 0:
            N = int(Y.size / (D + 2))
        else:
            print """
                Error: Number of samples in model output file must be a multiple of (D+2), 
                where D is the number of parameters in your parameter file.
                (You have calc_second_order set to false.)
              """
            exit()

    if conf_level < 0 or conf_level > 1:    
        print "Error: Confidence level must be between 0-1."
        exit() 
    
    A = np.empty([N])
    B = np.empty([N])
    C_A = np.empty([N, D])
    C_B = np.empty([N, D])
    Yindex = 0
    
    for i in range(N):
        A[i] = Y[Yindex]
        Yindex += 1
        
        for j in range(D):
            C_A[i][j] = Y[Yindex]
            Yindex += 1
        
        if calc_second_order:
            for j in range(D):
                C_B[i][j] = Y[Yindex]
                Yindex += 1
        
        B[i] = Y[Yindex]
        Yindex += 1
    
    # First order (+conf.) and Total order (+conf.)
    Si = dict((k, [None]*D) for k in ['S1','S1_conf','ST','ST_conf'])
    if print_to_console:
        print "Parameter First_Order First_Order_Conf Total_Order Total_Order_Conf"
    a0, a1, a2 = [np.empty([N]) for _ in xrange(3)]

    for j in range(D):
        for i in range(N):
            a0[i] = A[i]
            a1[i] = C_A[i][j]
            a2[i] = B[i]
            
        Si['S1'][j] = compute_first_order(a0, a1, a2, N)
        Si['S1_conf'][j] = compute_first_order_confidence(a0, a1, a2, N, num_resamples, conf_level)
        Si['ST'][j] = compute_total_order(a0, a1, a2, N)
        Si['ST_conf'][j] = compute_total_order_confidence(a0, a1, a2, N, num_resamples, conf_level)
        
        if print_to_console:
            print "%s %f %f %f %f" % (param_file['names'][j], Si['S1'][j], Si['S1_conf'][j], Si['ST'][j], Si['ST_conf'][j])
    
    # Second order (+conf.)
    if calc_second_order:
        if print_to_console:
            print "\nParameter_1 Parameter_2 Second_Order Second_Order_Conf"
        a0, a1, a2, a3, a4 = [np.empty([N]) for _ in xrange(5)]
        
        for j in range(D):
            for k in range(j+1, D):
                
                for i in range(N):
                    a0[i] = A[i]
                    a1[i] = C_B[i][j]
                    a2[i] = C_A[i][k]
                    a3[i] = C_A[i][j]
                    a4[i] = B[i]
                    
                S2 = compute_second_order(a0, a1, a2, a3, a4, N)
                S2c = compute_second_order_confidence(a0, a1, a2, a3, a4, N, num_resamples, conf_level)
                
                if print_to_console:
                    print "%s %s %f %f" % (param_file['names'][j], param_file['names'][k], S2, S2c)                        
    
    return Si            
        
def compute_first_order(a0, a1, a2, N):

    c = np.mean(a0)
    EY2 = np.mean((a0-c)*(a2-c))
    V = np.sum((a2-c)**2)/(N-1) - np.mean(a2-c)**2
    U = np.sum((a1-c)*(a2-c))/(N-1)

    return (U - EY2) / V

def compute_first_order_confidence(a0, a1, a2, N, num_resamples, conf_level):
    
    s  = np.empty([num_resamples])
    
    for i in range(num_resamples):        
        r = np.random.randint(N, size=N)        
        s[i] = compute_first_order(a0[r], a1[r], a2[r], N)
    
    return norm.ppf(0.5 + conf_level/2) * s.std(ddof=1)

def compute_total_order(a0, a1, a2, N):
    
    c = np.average(a0)
    EY2 = np.mean(a0-c)**2
    V = np.sum((a0-c)**2)/(N-1) - EY2
    U = np.sum((a0-c)*(a1-c))/(N-1)

    return (1 - (U-EY2) / V)

def compute_total_order_confidence(a0, a1, a2, N, num_resamples, conf_level):
    
    s  = np.empty([num_resamples])

    if conf_level < 0 or conf_level > 1:    
        print "Error: Confidence level must be between 0-1."
        exit() 
    
    for i in range(num_resamples):
        r = np.random.randint(N, size=N)  
        s[i] = compute_total_order(a0[r], a1[r], a2[r], N)
    
    return norm.ppf(0.5 + conf_level/2) * s.std(ddof=1)

def compute_second_order(a0, a1, a2, a3, a4, N):
    
    c = np.average(a0)    
    EY = np.mean((a0-c)*(a4-c))
    EY2 = np.mean((a1-c)*(a3-c))
    
    V = np.sum((a1-c)**2)/(N-1) - np.mean(a1-c)**2
    Vij = np.sum((a1-c)*(a2-c))/(N-1) - EY2
    Vi = np.sum((a2-c)*(a4-c))/(N-1) - EY
    Vj = np.sum((a3-c)*(a4-c))/(N-1) - EY2
    
    return (Vij - Vi - Vj) / V

def compute_second_order_confidence(a0, a1, a2, a3, a4, N, num_resamples, conf_level):
    
    s  = np.empty([num_resamples])

    if conf_level < 0 or conf_level > 1:    
        print "Error: Confidence level must be between 0-1."
        exit() 
    
    for i in range(num_resamples):
        r = np.random.randint(N, size=N)
        s[i] = compute_second_order(a0[r], a1[r], a2[r], a3[r], a4[r], N)
    
    return norm.ppf(0.5 + conf_level/2) * s.std(ddof=1)

if __name__ == "__main__":

    parser = common_args.create()
    parser.add_argument('--max-order', type=int, required=False, default=2, choices=[1, 2], help='Maximum order of sensitivity indices to calculate')
    parser.add_argument('-r', '--resamples', type=int, required=False, default=1000, help='Number of bootstrap resamples for Sobol confidence intervals')
    args = parser.parse_args()

    analyze(args.paramfile, args.model_output_file, args.column, (args.max_order == 2), num_resamples = args.resamples, delim = args.delimiter, print_to_console=True)
