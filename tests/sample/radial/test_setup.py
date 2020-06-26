import numpy as np
from SALib.test_functions import Sobol_G

def setup_G4():
    problem = {
        'num_vars': 20,
        'names': ['x{}'.format(i) for i in range(1, 21)],
        'bounds': [[0, 1] * 20]
    }

    a = np.array([100, 0, 100, 100, 100,
                  100, 1, 0, 100, 100,
                  0, 100, 100, 100, 1,
                  100, 100, 0, 100, 1])

    alpha = np.array([1, 4, 1, 1, 1,
                      1, 0.5, 3, 1, 1,
                      2, 1, 1, 1, 0.5,
                      1, 1, 1.5, 1, 0.5])

    analytic_vals = Sobol_G._calc_analytic(a, alpha, 20)

    return problem, alpha, a, analytic_vals


def setup_G10():
    problem = {
        'num_vars': 20,
        'names': ['x{}'.format(i) for i in range(1, 21)],
        'bounds': [[0, 1] * 20]
    }

    a = np.array([100, 0, 100, 100, 100, 
              100, 0, 1, 0, 0, 
              1, 0, 100, 100, 0,
              100, 100, 3, 100, 0])

    alpha = np.array([1, 4, 1, 1, 1,
                    1, 0.4, 3, 0.8, 0.7,
                    2, 1.3, 1, 1, 0.5,
                    1, 1, 2.5, 1, 0.6])

    analytic_vals = Sobol_G._calc_analytic(a, alpha, 20)

    return problem, alpha, a, analytic_vals