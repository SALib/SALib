from numpy.testing import assert_allclose
import numpy as np

from SALib.sample.sobol_corr import sample
from SALib.analyze.sobol_corr import analyze
from .utils import get_sensitivity_stats



def make_problem1():

    def func(v):
        return np.sum(v, axis=1) # f(x) = x1+x2+x3

    
    def si_analytical():
        return {
            'S1_ind':  [0.02, 0.05, 0.03],
            'ST_ind':  [0.02, 0.05, 0.03],
            'S1_full': [0.95, 0.40, 0.60],
            'ST_full': [0.95, 0.40, 0.60],
        }

    problem = {
        'num_vars': 3,
        'names': ['x1', 'x2', 'x3'],
        'distrs': ['norm', 'norm', 'norm'],
        'bounds': [
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        'corr': [
            [1.0, 0.5, 0.8],
            [0.5, 1.0, 0.0],
            [0.8, 0.0, 1.0],
        ],
        'func': func,
        'analytical': si_analytical,
    }
    
    return problem


def make_problem2():

    def func(v):
        return np.sum(v, axis=1) # f(x) = x1+x2+x3

    def si_analytical():
        return {
            'S1_ind':  [0.70, 0.37, 0.50],
            'ST_ind':  [0.70, 0.37, 0.50],
            'S1_full': [0.49, 0.05, 0.25],
            'ST_full': [0.49, 0.05, 0.25],
        }

    problem = {
        'num_vars': 3,
        'names': ['x1', 'x2', 'x3'],
        'distrs': ['norm', 'norm', 'norm'],
        'bounds': [
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        'corr': [
            [1.0, -0.5, 0.2],
            [-0.5, 1.0, -0.7],
            [0.2, -0.7, 1.0],
        ],
        'func': func,
        'analytical': si_analytical,
    }
    
    return problem


def sobol_corr_Si(problem):
    sample_args = {
        'n_sample': 1000,
    }

    analyze_args = {
        **sample_args,
        'n_boot': 100,
        'estimator': 'soboleff2',
    }
    
    x = sample(problem, **sample_args)
    y = problem['func'](x)
    return analyze(problem, y, **analyze_args)


def test_analytical1():
    problem = make_problem1()
    results_c = get_sensitivity_stats(problem, sobol_corr_Si, n=500)
    results_a = problem['analytical']()
    assert_allclose(results_c['S1_ind'], results_a['S1_ind'], atol=0.05, rtol=0)
    assert_allclose(results_c['ST_ind'], results_a['ST_ind'], atol=0.05, rtol=0)
    assert_allclose(results_c['S1_full'], results_a['S1_full'], atol=0.05, rtol=0)
    assert_allclose(results_c['ST_full'], results_a['ST_full'], atol=0.05, rtol=0)


def test_analytical2():
    problem = make_problem2()
    results_c = get_sensitivity_stats(problem, sobol_corr_Si, n=500)
    results_a = problem['analytical']()
    assert_allclose(results_c['S1_ind'], results_a['S1_ind'], atol=0.2, rtol=0)
    assert_allclose(results_c['ST_ind'], results_a['ST_ind'], atol=0.2, rtol=0)
    assert_allclose(results_c['S1_full'], results_a['S1_full'], atol=0.2, rtol=0)
    assert_allclose(results_c['ST_full'], results_a['ST_full'], atol=0.2, rtol=0)
