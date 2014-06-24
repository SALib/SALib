from __future__ import division
from ..util import read_param_file
from sys import exit
import numpy as np

# Perform Morris Analysis on file of model results
#returns a dictionary with keys equal to the parameter labels and values of numpy arrays containing
#Mu Sigma Mu_Star Mu_Star_Conf in that order.
def analyze(pfile, input_file, output_file, column = 0, delim = ' ', num_resamples = 1000,conf_level=0.95):
    
    param_file = read_param_file(pfile)
    Y = np.loadtxt(output_file, delimiter = delim)
    X = np.loadtxt(input_file, delimiter = delim)

    if len(Y.shape) == 1: Y = Y.reshape((len(Y),1))
    if len(X.shape) == 1: X = X.reshape((len(X),1))
    
    if Y.ndim > 1:
        Y = Y[:, column]
    
    D = param_file['num_vars']
    
    if Y.size % (D+1) == 0:    
        N = int(Y.size / (D + 1))
    else:
        print """
                Error: Number of samples in model output file must be a multiple of (D+1), 
                where D is the number of parameters in your parameter file.
              """
        exit()            
    
    ee = np.empty([N, D])
    
    # For each of the N trajectories
    for i in range(N):
        
        # Set up the indices corresponding to this trajectory
        j = np.arange(D+1) + i*(D + 1)
        j1 = j[0:D]
        j2 = j[1:D+1]
        
        # The elementary effect is (change in output)/(change in input)
        # Each parameter has one EE per trajectory, because it is only changed once in each trajectory
        ee[i,:] = np.linalg.solve((X[j2,:] - X[j1,:]), Y[j2] - Y[j1]) 
    
    # Output the Mu, Mu*, and Sigma Values
    sens={}
    print "Parameter Mu Sigma Mu_Star Mu_Star_Conf"
    for j in range(D):
        mu = np.average(ee[:,j])
        mu_star = np.average(np.abs(ee[:,j]))
        sigma = np.std(ee[:,j])
        mu_star_conf = compute_mu_star_confidence(ee[:,j], N, num_resamples,conf_level)
        
        print "%s %f %f %f %f" % (param_file['names'][j], mu, sigma, mu_star, mu_star_conf)
        sens[param_file['names'][j]]=np.r_[mu,sigma,mu_star,mu_star_conf]

    return sens
        

def compute_mu_star_confidence(ee, N, num_resamples,conf_level):

    if conf_level==0.99:
        zcrit=2.575
    elif conf_level==0.98:
        zcrit=2.33
    elif conf_level==0.95:
        zcrit=1.96
    elif conf_level==0.90:
        zcrit=1.645
    else:
        raise Exception("Confidence level can be .9, .95, .98, .99")
   
    ee_resampled = np.empty([N])
    mu_star_resampled  = np.empty([num_resamples])

    for i in range(num_resamples):
       for j in range(N):
           
           index = np.random.randint(0, N)
           ee_resampled[j] = ee[index]
       
       mu_star_resampled[i] = np.average(np.abs(ee_resampled))

    return zcrit * mu_star_resampled.std(ddof=1)
    
        
