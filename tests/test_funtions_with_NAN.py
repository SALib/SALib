# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 09:37:30 2019

@author: jchaconhurtado
"""
import sys
sys.path.insert(0, '..\src')


import numpy as np
from SALib.sample import saltelli
from SALib.analyze import delta, rbd_fast, sobol, morris, ff, fast, dgsm
import pytest

def setup_samples(N=1000):
    def _nan_func(x):
        if x[0]/x[1] > 1:
            out = np.nan
        else:
            out = np.sum(x)
            
            
        return out
    problem = dict(num_vars = 2,
                   names = ['x', 'y'],
                   bounds = [[0,1], [0,1]],
                   )
    
    param_values = saltelli.sample(problem, N)
    model_results = np.array([_nan_func(Xi) for Xi in param_values])
    return problem, model_results, param_values

def test_Sobol_with_NAN():
    '''
    Test if sobol.analyze can calculate S1 and ST with NAN values in the model 
    results.
    '''
    problem, model_results, _ = setup_samples()
    out = sobol.analyze(problem, model_results, ignore_nans=True)
    assert np.all(np.isfinite(out['S1'])), 'S1 index has NAN values'
    assert np.all(np.isfinite(out['ST'])), 'ST index has NAN values'

def test_Sobol_with_no_NAN_flag():
    '''
    Test if sobol.analyze raise ValueError when nan are passed with ignore_nans 
    flag is set to False
    '''
    problem, model_results, _ = setup_samples()
    
    #  Should raise a ValueError type of error
    with pytest.raises(ValueError):
        sobol.analyze(problem, model_results, ignore_nans=False)

def test_Sobol_with_no_NAN_flag_as_default():
    '''
    Test if sobol.analyze raise ValueError when nan are passed with ignore_nans 
    flag is set to False as default value
    '''
    problem, model_results, _ = setup_samples()
    
    #  Should raise a ValueError type of error
    with pytest.raises(ValueError):
        sobol.analyze(problem, model_results)

def test_Delta_with_NAN():
    '''
    Test if delta.analyze raise a ValueError when nan are passed in the Y 
    values
    '''
    problem, model_results, param_values = setup_samples()
    
    #  Should raise a ValueError type of error
    with pytest.raises(ValueError):
        delta.analyze(problem, param_values, model_results)

def test_Rbd_fast_with_NAN():
    '''
    Test if rbd_fast.analyze raise a ValueError when nan are passed in the Y 
    values
    '''
    problem, model_results, param_values = setup_samples()
    
    #  Should raise a ValueError type of error
    with pytest.raises(ValueError):
        rbd_fast.analyze(problem, model_results, param_values)

def test_Morris_with_NAN():
    '''
    Test if morris.analyze raise a ValueError when nan are passed in the Y 
    values
    '''
    problem, model_results, param_values = setup_samples()
    
    #  Should raise a ValueError type of error
    with pytest.raises(ValueError):
        morris.analyze(problem, param_values, model_results)

def test_FF_with_NAN():
    '''
    Test if ff.analyze raise a ValueError when nan are passed in the Y values
    '''
    problem, model_results, param_values = setup_samples()
    
    #  Should raise a ValueError type of error
    with pytest.raises(ValueError):
        ff.analyze(problem, param_values, model_results)

def test_Fast_with_NAN():
    '''
    Test if fast.analyze raise a ValueError when nan are passed in the Y values
    '''
    problem, model_results, _ = setup_samples()
    
    #  Should raise a ValueError type of error
    with pytest.raises(ValueError):
        fast.analyze(problem, model_results)        

def test_Dgsm_with_NAN():
    '''
    Test if dgsm.analyze raise a ValueError when nan are passed in the Y values
    '''
    problem, model_results, param_values = setup_samples()
    
    #  Should raise a ValueError type of error
    with pytest.raises(ValueError):
        dgsm.analyze(problem, param_values, model_results)        

# if __name__ == '__main__':
#     test_Sobol_with_NAN()    
#     test_Sobol_with_no_NAN_flag()    
#     test_Sobol_with_no_NAN_flag_as_default()
#     test_Delta_with_NAN()
#     test_rbd_fast_with_NAN()        
#     test_Morris_with_NAN()
#     test_FF_with_NAN()        
#     test_Fast_with_NAN()                    
#     test_Dgsm_with_NAN()  