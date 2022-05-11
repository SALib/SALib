from src.SALib.sample import saltelli
from src.SALib.analyze import sobol
from src.SALib.test_functions import Ishigami, Sobol_G
import numpy as np
import pandas as pd
from numpy.testing import assert_allclose

def main():
    # problem = {
    #     'num_vars': 3,
    #     'names': ['x1', 'x2', 'x3'],
    #     'bounds': [
    #         [-3.14159265359, 3.14159265359],
    #         [-3.14159265359, 3.14159265359],
    #         [-3.14159265359, 3.14159265359]
    #     ]
    # }
    # param_values = saltelli.sample(problem, 1024)
    # Y = Ishigami.evaluate(param_values)
    # Si = sobol.analyze(problem, Y, print_to_console=True)

    problem = {'num_vars': 6,
               'names': ['x1', 'x2', 'x3', 'x4', 'x5', 'x6'],
               'bounds': [[0, 1], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]]}
    N = 4096
    a = np.array([78, 12, 0.5, 2, 97, 33])
    param_values = saltelli.sample(problem, N, calc_second_order=False)
    model_results = Sobol_G.evaluate(param_values, a)
    Si = sobol.analyze(problem, model_results, calc_second_order=False, print_to_console=True)

    expected = Sobol_G.sensitivity_index(a)
    print(pd.DataFrame({ 'Expected S1': expected }, problem['names']))
    assert_allclose(Si['S1'], expected, atol=1e-2, rtol=1e-6)
    assert_allclose(Si['S1_3inputs'], expected, atol=1e-2, rtol=1e-6)
    

if __name__ == "__main__":
    main()