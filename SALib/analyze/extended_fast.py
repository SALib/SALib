from __future__ import division
from ..util import read_param_file
from sys import exit
import numpy as np
import math

# Perform FAST Analysis on file of model results
# Returns a dictionary with keys 'S1' and 'ST'
# Where each entry is a list of size D (the number of parameters)
# Containing the indices in the same order as the parameter file
def analyze(pfile, output_file, column = 0, M = 4, num_resamples = 1000, delim = ' '):
    
    param_file = read_param_file(pfile)
    Y = np.loadtxt(output_file, delimiter = delim)

    if len(Y.shape) == 1: Y = Y.reshape((len(Y),1))

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
    Si = dict((k, [None]*D) for k in ['S1','ST'])
    for i in range(D):
        l = range(i*N, (i+1)*N)
        Si['S1'][i] = compute_first_order(Y[l], N, M, omega[0])
        Si['ST'][i] = compute_total_order(Y[l], N, omega[0])        
        print "%s %f %f" % (param_file['names'][i], Si['S1'][i], Si['ST'][i])
    return Si
    
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
