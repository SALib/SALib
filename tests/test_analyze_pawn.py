from SALib.util import read_param_file
from SALib import ProblemSpec
from SALib.test_functions import Ishigami


def test_analyze_pawn():
    param_file = "src/SALib/test_functions/params/Ishigami_groups.txt"
    problem = read_param_file(param_file)
    sp = ProblemSpec(problem)
    sp.sample_sobol(512).evaluate(Ishigami.evaluate).analyze_pawn()
