from SALib import ProblemSpec
from SALib.test_functions import lake_problem
import random


def test_odd_sample_size():
    """Specific regression test to ensure odd number of samples are handled."""
    sp = ProblemSpec(
        {
            "names": ["a", "q", "b", "mean", "stdev", "delta", "alpha"],
            "bounds": [
                [0.0, 0.1],
                [2.0, 4.5],
                [0.1, 0.45],
                [0.01, 0.05],
                [0.001, 0.005],
                [0.93, 0.99],
                [0.2, 0.5],
            ],
            "outputs": ["max_P", "Utility", "Inertia", "Reliability"],
        }
    )

    # eFAST analysis with a series of odd sample sizes
    # These should all pass and not raise IndexError.
    (sp.sample_fast(511).evaluate(lake_problem.evaluate).analyze_fast())

    (sp.sample_fast(1313).evaluate(lake_problem.evaluate).analyze_fast())

    odds = [random.randrange(5099, 6500, 2) for _ in range(0, 3)]

    for No in odds:
        (sp.sample_fast(No).evaluate(lake_problem.evaluate).analyze_fast())


def test_even_sample_size():
    """Specific regression test to ensure odd number of samples are handled."""
    sp = ProblemSpec(
        {
            "names": ["a", "q", "b", "mean", "stdev", "delta", "alpha"],
            "bounds": [
                [0.0, 0.1],
                [2.0, 4.5],
                [0.1, 0.45],
                [0.01, 0.05],
                [0.001, 0.005],
                [0.93, 0.99],
                [0.2, 0.5],
            ],
            "outputs": ["max_P", "Utility", "Inertia", "Reliability"],
        }
    )

    # Test specific even number
    (sp.sample_fast(1024).evaluate(lake_problem.evaluate).analyze_fast())

    evens = [random.randrange(5026, 6500, 2) for _ in range(0, 3)]

    for Ne in evens:
        (sp.sample_fast(Ne).evaluate(lake_problem.evaluate).analyze_fast())
