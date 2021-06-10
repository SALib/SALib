"""Example showing how to use the ProblemSpec approach.

Showcases method chaining, and parallel model runs using
all available processors.

The following caveats apply:

1. Functions passed into `sample`, `analyze`, `evaluate` and `evaluate_*` must 
   accept a numpy array of `X` values as the first parameter, and return a 
   numpy array of results.
2. Parallel evaluation is only beneficial for long-running models
3. Currently, model results must fit in memory - no on-disk caching is provided.
"""

from SALib.analyze import sobol
from SALib.sample import saltelli
from SALib.test_functions import Ishigami
from SALib import ProblemSpec
import numpy as np

import time


if __name__ == '__main__':

    # Create the SALib Problem specification
    sp = ProblemSpec({
        'names': ['x1', 'x2', 'x3'],
        'groups': None,
        'bounds': [[-np.pi, np.pi]]*3,
        'outputs': ['Y']
    })

    # Single core example
    start = time.perf_counter()
    (sp.sample_saltelli(25000)
        .evaluate(Ishigami.evaluate)
        .analyze_sobol(calc_second_order=True, conf_level=0.95))
    print("Time taken with 1 core:", time.perf_counter() - start, '\n')

    # Samples, model results and analyses can be extracted:
    # print(sp.samples)
    # print(sp.results)
    # print(sp.analysis)
    # print(sp.to_df())

    # Same above, but passing in specific functions
    # (sp.sample(saltelli.sample, 25000, calc_second_order=True)
    #     .evaluate(Ishigami.evaluate)
    #     .analyze(sobol.analyze, calc_second_order=True, conf_level=0.95))

    # Parallel example
    start = time.perf_counter()
    (sp.sample(saltelli.sample, 25000)
         # can specify number of processors to use with `nprocs`
        .evaluate_parallel(Ishigami.evaluate, nprocs=2)
        .analyze(sobol.analyze, calc_second_order=True, conf_level=0.95))
    print("Time taken with all available cores:", time.perf_counter() - start, '\n')

    print(sp)
    
    # Distributed example
    # Specify itself as servers as an example
    servers = ('localhost:55774',
               'localhost:55775',
               'localhost:55776')

    start = time.perf_counter()
    (sp.sample(saltelli.sample, 25000)
        .evaluate_distributed(Ishigami.evaluate, nprocs=2, servers=servers, verbose=True)
        .analyze(sobol.analyze, calc_second_order=True, conf_level=0.95))
    print("Time taken with distributed cores:", time.perf_counter() - start, '\n')

    print(sp)

    # To display plots:
    # import matplotlib.pyplot as plt
    # sp.plot()
    # plt.show()

# First-order indices expected with Saltelli sampling:
# x1: 0.3139
# x2: 0.4424
# x3: 0.0
