from SALib import ProblemSpec
from SALib.test_functions import lake_problem


def test_odd_sample_size():
    """Specific regression test to ensure odd number of samples are handled.
    """
    seed_val = 101

    sp = ProblemSpec({
        'names': ['a', 'q', 'b', 'mean', 'stdev', 'delta', 'alpha'],
        'bounds': [[0.0, 0.1], [2.0, 4.5], [0.1, 0.45], [0.01, 0.05],
                   [0.001, 0.005], [0.93, 0.99], [0.2, 0.5]],
        'outputs': ['max_P', 'Utility', 'Inertia', 'Reliability']
    })

    # eFAST analysis with a series of odd and even samples
    # These should all pass and not raise IndexError.
    (sp.sample_fast(511)
        .evaluate(lake_problem.evaluate)
        .analyze_fast())

    (sp.sample_fast(513)
        .evaluate(lake_problem.evaluate)
        .analyze_fast())

    (sp.sample_fast(1024)
        .evaluate(lake_problem.evaluate)
        .analyze_fast())

    (sp.sample_fast(1025)
        .evaluate(lake_problem.evaluate)
        .analyze_fast())

    (sp.sample_fast(4095)
        .evaluate(lake_problem.evaluate)
        .analyze_fast())
