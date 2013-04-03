from __future__ import division
from ..util import read_param_file
from sys import exit
import numpy as np
import math

# Perform FAST Analysis on file of model results
def analyze(pfile, output_file, column = 0, M = 4, num_resamples = 1000, delim = ' '):
    
    param_file = read_param_file(pfile)
    Y = np.loadtxt(output_file, delimiter = delim)
    D = param_file['num_vars']
    
    if Y.ndim > 1:
        Y = Y[:, column]
    
    if Y.size % (D) == 0:
        N = int(Y.size / D)
    else:
        print """
            Error: Number of samples in model output file must be a multiple of D, 
            where D is the number of parameters in your parameter file.
          """
        exit()
    
    # Recreate the vector omega used in the sampling
    omega = np.empty([D])
    omega[0] = math.floor((N - 1) / (2 * M))
    m = math.floor(omega[0] / (2 * M))
    
    if m >= (D-1):
        omega[1:] = np.floor(np.linspace(1, m, D-1)) 
    else:
        omega[1:] = np.arange(D-1) % m + 1
    
    # Calculate and Output the First and Total Order Values
    print "Parameter First Total"
    for i in range(D):
        l = range(i*N, (i+1)*N)
        first = compute_first_order(Y[l], N, M, omega[0])
        total = compute_total_order(Y[l], N, omega[0])        
        print "%s %f %f" % (param_file['names'][i], first, total)
    
def compute_first_order(outputs, N, M, omega):
    f = np.fft.fft(outputs)
    Sp = np.power(np.absolute(f[range(1,int(N/2))]) / N, 2)
    V = 2*np.sum(Sp)
    D1 = 2*np.sum(Sp[list(np.arange(1,M)*int(omega) - 1)])
    return D1/V

def compute_total_order(outputs, N, omega):
    f = np.fft.fft(outputs)
    Sp = np.power(np.absolute(f[range(1,int(N/2))]) / N, 2)
    V = 2*np.sum(Sp)
    Dt = 2*sum(Sp[range(int(omega/2))])
    return (1 - Dt/V)
    
#def compute_first_order_confidence(outputs, N, M, omega, num_resamples):
#    
#    output_resample = np.empty([N])
#    s  = np.empty([num_resamples])
#    
#    for i in range(num_resamples):
#        for j in range(N):
#            
#            index = np.random.randint(0, N)
#            output_resample[j] = outputs[index]
#            
#        s[i] = compute_first_order(output_resample, N, M, omega)
#    
#    return 1.96 * s.mean()#std(ddof=1)
#
#def compute_total_order_confidence(a0, a1, a2, N, num_resamples):
#    
#    b0 = np.empty([N])
#    b1 = np.empty([N])
#    b2 = np.empty([N])
#    s  = np.empty([num_resamples])
#    
#    for i in range(num_resamples):
#        for j in range(N):
#            
#            index = np.random.randint(0, N)
#            b0[j] = a0[index]
#            b1[j] = a1[index]
#            b2[j] = a2[index]
#        
#        s[i] = compute_total_order(b0, b1, b2, N)
#    
#    return 1.96 * s.std(ddof=1)
