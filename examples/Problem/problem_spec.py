"""Example showing how to use the ProblemSpec approach.

Showcases method chaining, and parallel model runs using
all available cores.
"""

from SALib.analyze import sobol
from SALib.sample import saltelli
from SALib.test_functions import Ishigami
from SALib import ProblemSpec

import time


if __name__ == '__main__':

    # Create the SALib Problem specification
    sp = ProblemSpec({
        'names': ['x1', 'x2', 'x3'],
        'groups': None,
        'bounds': [[-3.14159265359, 3.14159265359],
                [-3.14159265359, 3.14159265359],
                [-3.14159265359, 3.14159265359]],
        'outputs': ['Y']
    })

    # Single core example
    start = time.perf_counter()
    (sp.sample(saltelli.sample, 75000, calc_second_order=True)
        .evaluate(Ishigami.evaluate)
        .analyze(sobol.analyze, calc_second_order=True, conf_level=0.95))
    print("Time taken with 1 core:", time.perf_counter() - start, '\n')

    # Parallel example
    start = time.perf_counter()
    (sp.sample(saltelli.sample, 75000, calc_second_order=True)
         # can specify number of processors to use with `nprocs`
        .evaluate_parallel(Ishigami.evaluate)
        .analyze(sobol.analyze, calc_second_order=True, conf_level=0.95))
    print("Time taken with all available cores:", time.perf_counter() - start, '\n')

    print(sp)
    
    # Distributed example
    # Specify itself as servers as an example
    servers = ('localhost:55774',
               'localhost:55775',
               'localhost:55776')

    start = time.perf_counter()
    (sp.sample(saltelli.sample, 75000, calc_second_order=True)
        .evaluate_distributed(Ishigami.evaluate, nprocs=2, servers=servers, verbose=True)
        .analyze(sobol.analyze, calc_second_order=True, conf_level=0.95))
    print("Time taken with distributed cores:", time.perf_counter() - start, '\n')

    print(sp)

# First-order indices expected with Saltelli sampling:
# x1: 0.3139
# x2: 0.4424
# x3: 0.0
