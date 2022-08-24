"""Example showing how to use the ProblemSpec approach.

Showcases method chaining, and parallel model runs using 2 processors.

The following caveats apply:

1. Functions passed into `sample`, `analyze` and `evaluate` must
   accept a numpy array of `X` values as the first parameter, and return a
   numpy array of results.
2. Parallel evaluation/analysis is only beneficial for long-running models
   or large datasets
3. Currently, results must fit in memory - no on-disk caching is provided.
"""

from SALib.analyze import sobol
from SALib.sample import saltelli
from SALib.test_functions import Ishigami
from SALib import ProblemSpec
import numpy as np

import time


if __name__ == "__main__":

    # Create the SALib Problem specification
    sp = ProblemSpec(
        {
            "names": ["x1", "x2", "x3"],
            "groups": None,
            "bounds": [[-np.pi, np.pi]] * 3,
            "outputs": ["Y"],
        }
    )

    # Single core example
    start = time.perf_counter()
    (
        sp.sample_saltelli(2**15)
        .evaluate(Ishigami.evaluate)
        .analyze_sobol(calc_second_order=True, conf_level=0.95)
    )
    print("Time taken with 1 core:", time.perf_counter() - start, "\n")

    # Same above example, but passing in specific functions
    # (sp.sample(saltelli.sample, 25000, calc_second_order=True)
    #     .evaluate(Ishigami.evaluate)
    #     .analyze(sobol.analyze, calc_second_order=True, conf_level=0.95))

    # Samples, model results and analyses can be extracted:
    # print(sp.samples)
    # print(sp.results)
    # print(sp.analysis)
    # print(sp.to_df())

    # Can set pre-existing samples/results as needed
    # sp.samples = some_numpy_array
    # sp.set_samples(some_numpy_array)
    #
    # Using method chaining...
    # (sp.set_samples(some_numpy_array)
    #    .set_results(some_result_array)
    #    .analyze_sobol(calc_second_order=True, conf_level=0.95))

    # Parallel example
    start = time.perf_counter()
    (
        sp.sample(saltelli.sample, 2**15)
        # can specify number of processors to use with `nprocs`
        # this will be capped to the number of detected processors
        # or, in the case of analysis, the number of outputs.
        .evaluate(Ishigami.evaluate, nprocs=2).analyze_sobol(
            calc_second_order=True, conf_level=0.95, nprocs=2
        )
    )
    print("Time taken with 2 cores:", time.perf_counter() - start, "\n")

    print(sp)

    # Distributed example
    # Specify itself as servers as an example
    servers = ("localhost:55774", "localhost:55775", "localhost:55776")

    start = time.perf_counter()
    (
        sp.sample(saltelli.sample, 2**15)
        .evaluate_distributed(
            Ishigami.evaluate, nprocs=2, servers=servers, verbose=True
        )
        .analyze(sobol.analyze, calc_second_order=True, conf_level=0.95)
    )
    print("Time taken with distributed cores:", time.perf_counter() - start, "\n")

    print(sp)

    # To display plots:
    # import matplotlib.pyplot as plt
    # sp.plot()
    # plt.show()

# First-order indices expected with Saltelli sampling:
# x1: 0.3139
# x2: 0.4424
# x3: 0.0
