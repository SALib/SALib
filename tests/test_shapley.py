from numpy.testing import assert_allclose
import numpy as np

from SALib.sample.shapley import sample
from SALib.analyze.shapley import analyze
from .utils import get_sensitivity_stats



# test 1
def make_problem1(rho):
    betta = np.array([1.0, 1.0])
    mu = np.array([0.0, 0.0])
    sigma = np.array([1.0, 2.0])

    def func(v):
        return v.dot(np.expand_dims(betta, axis=1))

    def si_analytical(rho):
        sigma2 = betta[0]**2*sigma[0]**2+2*rho*betta[0]*betta[1]*sigma[0]*sigma[1]+betta[1]**2*sigma[1]**2
        Sh1 = (betta[0]**2*sigma[0]**2*(1-rho**2/2)+rho*betta[0]*betta[1]*sigma[0]*sigma[1]+betta[1]**2*sigma[1]**2*rho**2/2)/sigma2
        Sh2 = (betta[1]**2*sigma[1]**2*(1-rho**2/2)+rho*betta[0]*betta[1]*sigma[0]*sigma[1]+betta[0]**2*sigma[0]**2*rho**2/2)/sigma2
        S11 = (betta[0]**2*sigma[0]**2+2*rho*betta[0]*betta[1]*sigma[0]*sigma[1]+rho**2*betta[1]**2*sigma[1]**2)/sigma2
        S12 = (betta[1]**2*sigma[1]**2+2*rho*betta[0]*betta[1]*sigma[0]*sigma[1]+rho**2*betta[0]**2*sigma[0]**2)/sigma2
        ST1 = betta[0]**2*sigma[0]**2*(1-rho**2)/sigma2
        ST2 = betta[1]**2*sigma[1]**2*(1-rho**2)/sigma2

        return {
            'S1': [S11, S12],
            'ST': [ST1, ST2],
            'Sh': [Sh1, Sh2],
        }


    problem = {
        'num_vars': 2,
        'names': ['x1', 'x2'],
        'distrs': ['norm', 'norm'],
        'bounds': [
            [mu[0], sigma[0]],
            [mu[1], sigma[1]],
        ],
        'corr': [
            [1, rho],
            [rho, 1]
        ],
        'func': func,
        'analytical': si_analytical,
    }
    
    return problem

# test 2
def make_problem2(rho):
    betta = np.array([1.0, 1.0, 1.0])
    mu = np.array([0.0, 0.0, 0.0])
    sigma = np.array([1.0, 1.0, 2.0])

    def func(v):
        return v.dot(np.expand_dims(betta, axis=1))

    def si_analytical(rho):
        sigma2 = sum([betta[j]**2*sigma[j]**2 for j in range(3)])+2*rho*betta[1]*betta[2]*sigma[1]*sigma[2]
        Sh1 = betta[0]**2*sigma[0]**2/sigma2
        Sh2 = (betta[1]**2*sigma[1]**2+rho*betta[1]*betta[2]*sigma[1]*sigma[2]+rho**2/2*(betta[2]**2*sigma[2]**2-betta[1]**2*sigma[1]**2))/sigma2
        Sh3 = (betta[2]**2*sigma[2]**2+rho*betta[1]*betta[2]*sigma[1]*sigma[2]+rho**2/2*(betta[1]**2*sigma[1]**2-betta[2]**2*sigma[2]**2))/sigma2
        
        # only for these betta and sigma
        S11 = 1/(2+sigma2+2*rho*sigma2**0.5)
        S12 = (1+rho*sigma2**0.5)**2/(2+sigma2+2*rho*sigma2**0.5)
        S13 = (rho+sigma2**0.5)**2/(2+sigma2+2*rho*sigma2**0.5)
        ST1 = 1/(2+sigma2+2*rho*sigma2**0.5)
        ST2 = (1-rho**2)/(2+sigma2+2*rho*sigma2**0.5)
        ST3 = sigma2*(1-rho**2)/(2+sigma2+2*rho*sigma2**0.5)
        
        return {
            'S1': [S11, S12, S13],
            'ST': [ST1, ST2, ST3],
            'Sh': [Sh1, Sh2, Sh3],
        }


    problem = {
        'num_vars': 3,
        'names': ['x1', 'x2', 'x3'],
        'distrs': ['norm', 'norm', 'norm'],
        'bounds': [
            [mu[0], sigma[0]],
            [mu[1], sigma[1]],
            [mu[2], sigma[2]],
        ],
        'corr': [
            [1, 0.0, 0.0],
            [0.0, 1.0, rho],
            [0.0, rho, 1.0],
        ],
        'func': func,
        'analytical': si_analytical,
    }
    
    return problem


def shapley_Si(problem):
    analyze_args = {
        'n_perms': None,
        'n_var': 1000, # unstable results if too small
        'n_outer': 100,
        'n_inner': 100,
    }

    sample_args = {
        **analyze_args,
        'randomize': True
    }
    
    x = sample(problem, **sample_args)
    y = problem['func'](x)
    return analyze(problem, y, **analyze_args)


def atest_analytical1():
    rho = 0.5
    problem = make_problem1(rho)
    results_c = get_sensitivity_stats(problem, shapley_Si, n=200)
    results_a = problem['analytical'](rho)
    assert_allclose(results_c['S1'], results_a['S1'], atol=0.02, rtol=0)
    assert_allclose(results_c['ST'], results_a['ST'], atol=0.02, rtol=0)
    assert_allclose(results_c['Sh'], results_a['Sh'], atol=0.02, rtol=0)


def test_analytical2():
    rho = 0.5
    problem = make_problem2(rho)
    results_c = get_sensitivity_stats(problem, shapley_Si, n=200)
    results_a = problem['analytical'](rho)
    assert_allclose(results_c['S1'], results_a['S1'], atol=0.1, rtol=0)
    assert_allclose(results_c['ST'], results_a['ST'], atol=0.1, rtol=0)
    assert_allclose(results_c['Sh'], results_a['Sh'], atol=0.1, rtol=0)