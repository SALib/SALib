from SALib.analyze import sobol
from SALib.sample import saltelli
from SALib.test_functions import Ishigami
from SALib import ProblemSpec

import time

if __name__ == '__main__':

    sp = ProblemSpec({
        'names': ['x1', 'x2', 'x3'],
        'groups': None,
        'bounds': [[-3.14159265359, 3.14159265359],
                [-3.14159265359, 3.14159265359],
                [-3.14159265359, 3.14159265359]],
        'outputs': ['Y']
    })

    start = time.perf_counter()
    (sp.sample(saltelli.sample, 75000, calc_second_order=True)
        .run(Ishigami.evaluate)
        .analyze(sobol.analyze, calc_second_order=True, conf_level=0.95))
    print("Single:", time.perf_counter() - start)

    start = time.perf_counter()
    (sp.sample(saltelli.sample, 75000, calc_second_order=True)
        .run_parallel(Ishigami.evaluate, nprocs=3)
        .analyze(sobol.analyze, calc_second_order=True, conf_level=0.95))
    print("Multi:", time.perf_counter() - start)

    print(sp)

# First-order indices expected with Saltelli sampling:
# x1: 0.3139
# x2: 0.4424
# x3: 0.0
