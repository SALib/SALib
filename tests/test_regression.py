from pytest import fixture, mark

import numpy as np
from numpy.testing import assert_allclose

from SALib.analyze import delta, dgsm, fast, rbd_fast, sobol, morris, hdmr
from SALib.sample import fast_sampler, finite_diff, latin, saltelli
from SALib.sample.morris import sample as morris_sampler

from SALib.test_functions import Ishigami
from SALib.test_functions import linear_model_1
from SALib.test_functions import linear_model_2
from SALib.util import read_param_file, handle_seed


@fixture(scope="function")
def set_seed():
    """Sets seeds for random generators so that tests can be repeated

    It is necessary to set seeds for both the numpy.random, and
    the stdlib.random libraries.
    """
    rng = handle_seed(12345)

    # ensure we also keep the old stuff for the analyze functions that have not shifted over to rng yet
    seed = 123456
    np.random.seed(seed)

    return rng


class TestMorris:
    def test_regression_morris_vanilla(self, set_seed):
        """Note that this is a poor estimate of the Ishigami
        function.
        """
        rng = set_seed
        param_file = "src/SALib/test_functions/params/Ishigami.txt"
        problem = read_param_file(param_file)
        param_values = morris_sampler(problem, 10000, num_levels=4, optimal_trajectories=None, seed=rng)

        Y = Ishigami.evaluate(param_values)

        Si = morris.analyze(
            problem,
            param_values,
            Y,
            conf_level=0.95,
            print_to_console=False,
            num_levels=4,
            seed=rng
        )

        assert_allclose(Si["mu_star"], [7.682808, 7.875, 6.256295], atol=0, rtol=1e-5)

    def test_regression_morris_scaled(self, set_seed):
        """Note that this is a poor estimate of the Ishigami
        function.
        """
        rng = set_seed
        param_file = "src/SALib/test_functions/params/Ishigami.txt"
        problem = read_param_file(param_file)
        param_values = morris_sampler(problem, 10000, 4, optimal_trajectories=None, seed=rng)

        Y = Ishigami.evaluate(param_values)

        Si = morris.analyze(
            problem,
            param_values,
            Y,
            conf_level=0.95,
            scaled=True,
            print_to_console=False,
            num_levels=4,
            seed=rng
        )

        assert_allclose(Si["mu_star"], [0.540244, 0.657467, 0.433446], atol=0, rtol=1e-5)

    def test_regression_morris_groups(self, set_seed):
        rng = set_seed
        param_file = "src/SALib/test_functions/params/Ishigami_groups.txt"
        problem = read_param_file(param_file)

        param_values = morris_sampler(
            problem=problem, N=10000, num_levels=4, optimal_trajectories=None, seed=rng
        )

        Y = Ishigami.evaluate(param_values)

        Si = morris.analyze(
            problem,
            param_values,
            Y,
            conf_level=0.95,
            print_to_console=False,
            num_levels=4,
            seed=rng
        )

        assert_allclose(Si["mu_star"], [7.771541, 10.188284], atol=0, rtol=1e-5)

    def test_regression_morris_groups_brute_optim(self, set_seed):
        rng = set_seed
        param_file = "src/SALib/test_functions/params/Ishigami_groups.txt"
        problem = read_param_file(param_file)

        param_values = morris_sampler(
            problem=problem,
            N=50,
            num_levels=4,
            optimal_trajectories=6,
            local_optimization=False,
            seed=rng
        )

        Y = Ishigami.evaluate(param_values)

        Si = morris.analyze(
            problem,
            param_values,
            Y,
            conf_level=0.95,
            print_to_console=False,
            num_levels=4,
            seed=rng
        )

        assert_allclose(Si["mu"], [9.786986e+00, 1.776357e-15], atol=0, rtol=1e-5)

        assert_allclose(Si["sigma"], [6.453729, np.nan], atol=0, rtol=1e-5)

        assert_allclose(Si["mu_star"], [9.786986, 7.875], atol=0, rtol=1e-5)

    def test_regression_morris_groups_local_optim(self, set_seed):
        rng = set_seed
        param_file = "src/SALib/test_functions/params/Ishigami_groups.txt"
        problem = read_param_file(param_file)

        param_values = morris_sampler(
            problem=problem,
            N=500,
            num_levels=4,
            optimal_trajectories=20,
            local_optimization=True,
            seed=rng
        )

        Y = Ishigami.evaluate(param_values)

        Si = morris.analyze(
            problem,
            param_values,
            Y,
            conf_level=0.95,
            print_to_console=False,
            num_levels=4,
            seed=rng
        )

        assert_allclose(Si["mu_star"], [13.95285, 7.875], rtol=1e-5)

    def test_regression_morris_optimal(self, set_seed):
        """
        Tests the use of optimal trajectories with Morris.

        Uses brute force approach

        Note that the relative tolerance is set to a very high value
        (default is 1e-05) due to the coarse nature of the num_levels.
        """
        rng = set_seed
        param_file = "src/SALib/test_functions/params/Ishigami.txt"
        problem = read_param_file(param_file)
        param_values = morris_sampler(
            problem=problem,
            N=20,
            num_levels=4,
            optimal_trajectories=9,
            local_optimization=True,
            seed=rng
        )

        Y = Ishigami.evaluate(param_values)

        Si = morris.analyze(
            problem,
            param_values,
            Y,
            conf_level=0.95,
            print_to_console=False,
            num_levels=4,
            seed=rng
        )

        assert_allclose(
            Si["mu_star"], [11.175608,  7.875   ,  4.165864], atol=0, rtol=1e-5
        )


@mark.filterwarnings("ignore::UserWarning")
def test_regression_sobol():
    param_file = "src/SALib/test_functions/params/Ishigami.txt"
    problem = read_param_file(param_file)
    param_values = saltelli.sample(problem, 10000, calc_second_order=True)

    Y = Ishigami.evaluate(param_values)

    Si = sobol.analyze(
        problem, Y, calc_second_order=True, conf_level=0.95, print_to_console=False
    )

    assert_allclose(Si["S1"], [0.31, 0.44, 0.00], atol=5e-2, rtol=1e-1)
    assert_allclose(Si["ST"], [0.55, 0.44, 0.24], atol=5e-2, rtol=1e-1)
    assert_allclose(
        [Si["S2"][0][1], Si["S2"][0][2], Si["S2"][1][2]],
        [0.00, 0.25, 0.00],
        atol=5e-2,
        rtol=1e-1,
    )


@mark.filterwarnings("ignore::UserWarning")
def test_regression_sobol_parallel():
    param_file = "src/SALib/test_functions/params/Ishigami.txt"
    problem = read_param_file(param_file)
    param_values = saltelli.sample(problem, 10000, calc_second_order=True)

    Y = Ishigami.evaluate(param_values)

    Si = sobol.analyze(
        problem,
        Y,
        calc_second_order=True,
        parallel=True,
        conf_level=0.95,
        print_to_console=False,
    )

    assert_allclose(Si["S1"], [0.31, 0.44, 0.00], atol=5e-2, rtol=1e-1)
    assert_allclose(Si["ST"], [0.55, 0.44, 0.24], atol=5e-2, rtol=1e-1)
    assert_allclose(
        [Si["S2"][0][1], Si["S2"][0][2], Si["S2"][1][2]],
        [0.00, 0.25, 0.00],
        atol=5e-2,
        rtol=1e-1,
    )


@mark.filterwarnings("ignore::UserWarning")
def test_regression_sobol_groups():
    problem = {
        "num_vars": 3,
        "names": ["x1", "x2", "x3"],
        "bounds": [[-np.pi, np.pi]] * 3,
        "groups": ["G1", "G2", "G1"],
    }
    param_values = saltelli.sample(problem, 10000, calc_second_order=True)

    Y = Ishigami.evaluate(param_values)
    Si = sobol.analyze(
        problem,
        Y,
        calc_second_order=True,
        parallel=True,
        conf_level=0.95,
        print_to_console=False,
    )

    assert_allclose(Si["S1"], [0.55, 0.44], atol=5e-2, rtol=1e-1)
    assert_allclose(Si["ST"], [0.55, 0.44], atol=5e-2, rtol=1e-1)
    assert_allclose(Si["S2"][0][1], [0.00], atol=5e-2, rtol=1e-1)


@mark.filterwarnings("ignore::UserWarning")
def test_regression_sobol_groups_dists():
    problem = {
        "num_vars": 3,
        "names": ["x1", "x2", "x3"],
        "bounds": [[-np.pi, np.pi], [1.0, 0.2], [0.0, 3, 0.5]],
        "groups": ["G1", "G2", "G1"],
        "dists": ["unif", "lognorm", "triang"],
    }
    param_values = saltelli.sample(problem, 10000, calc_second_order=True)

    Y = Ishigami.evaluate(param_values)
    Si = sobol.analyze(
        problem,
        Y,
        calc_second_order=True,
        parallel=True,
        conf_level=0.95,
        print_to_console=False,
    )

    assert_allclose(Si["S1"], [0.427, 0.573], atol=5e-2, rtol=1e-1)
    assert_allclose(Si["ST"], [0.428, 0.573], atol=5e-2, rtol=1e-1)
    assert_allclose(Si["S2"][0][1], [0.001], atol=5e-2, rtol=1e-1)


def test_regression_fast():
    param_file = "src/SALib/test_functions/params/Ishigami.txt"
    problem = read_param_file(param_file)
    param_values = fast_sampler.sample(problem, 10000)

    Y = Ishigami.evaluate(param_values)

    Si = fast.analyze(problem, Y, print_to_console=False)
    assert_allclose(Si["S1"], [0.31, 0.44, 0.00], atol=5e-2, rtol=1e-1)
    assert_allclose(Si["ST"], [0.55, 0.44, 0.24], atol=5e-2, rtol=1e-1)


def test_regression_hdmr_ishigami():
    param_file = "src/SALib/test_functions/params/Ishigami.txt"
    problem = read_param_file(param_file)
    X = latin.sample(problem, 10000)
    Y = Ishigami.evaluate(X)
    options = {
        "maxorder": 2,
        "maxiter": 100,
        "m": 4,
        "K": 1,
        "R": 10000,
        "alpha": 0.95,
        "lambdax": 0.01,
        "print_to_console": False,
    }
    Si = hdmr.analyze(problem, X, Y, **options)
    assert_allclose(
        Si["Sa"][0 : problem["num_vars"]], [0.31, 0.44, 0.00], atol=5e-2, rtol=1e-1
    )
    assert_allclose(
        Si["ST"][0 : problem["num_vars"]], [0.55, 0.44, 0.24], atol=5e-2, rtol=1e-1
    )


def test_regression_hdmr_case1():
    problem = {
        "num_vars": 5,
        "names": ["x1", "x2", "x3", "x4", "x5"],
        "bounds": [[0, 1] * 5],
    }
    X = latin.sample(problem, 10000)
    Y = linear_model_1.evaluate(X)
    options = {
        "maxorder": 2,
        "maxiter": 100,
        "m": 2,
        "K": 1,
        "R": 10000,
        "alpha": 0.95,
        "lambdax": 0.01,
        "print_to_console": False,
    }
    Si = hdmr.analyze(problem, X, Y, **options)
    assert_allclose(Si["Sa"][0 : problem["num_vars"]], [0.20] * 5, atol=5e-2, rtol=1e-1)
    assert_allclose(Si["ST"][0 : problem["num_vars"]], [0.20] * 5, atol=5e-2, rtol=1e-1)


def test_regression_hdmr_case2():
    problem = {
        "num_vars": 5,
        "names": ["x1", "x2", "x3", "x4", "x5"],
        "bounds": [[0, 1] * 5],
    }
    # Generate correlated samples
    mean = np.zeros((problem["num_vars"]))
    cov = [
        [1, 0.6, 0.2, 0, 0],
        [0.6, 1, 0.2, 0, 0],
        [0.2, 0.2, 1, 0, 0],
        [0, 0, 0, 1, 0.2],
        [0, 0, 0, 0.2, 1],
    ]
    X = np.random.multivariate_normal(mean, cov, 10000)
    Y = linear_model_1.evaluate(X)
    options = {
        "maxorder": 2,
        "maxiter": 100,
        "m": 2,
        "K": 1,
        "R": 10000,
        "alpha": 0.95,
        "lambdax": 0.01,
        "print_to_console": False,
    }
    Si = hdmr.analyze(problem, X, Y, **options)
    assert_allclose(Si["Sa"][0 : problem["num_vars"]], [0.13] * 5, atol=5e-2, rtol=1e-1)
    assert_allclose(
        Si["Sb"][0 : problem["num_vars"]],
        [0.11, 0.11, 0.06, 0.03, 0.03],
        atol=5e-2,
        rtol=1e-1,
    )


def test_regression_hdmr_case3():
    problem = {
        "num_vars": 5,
        "names": ["x1", "x2", "x3", "x4", "x5"],
        "bounds": [[0, 1] * 5],
    }
    # Generate correlated samples
    mean = np.zeros((problem["num_vars"]))
    cov = [
        [1, 0.6, 0.2, 0, 0],
        [0.6, 1, 0.2, 0, 0],
        [0.2, 0.2, 1, 0, 0],
        [0, 0, 0, 1, 0.2],
        [0, 0, 0, 0.2, 1],
    ]
    X = np.random.multivariate_normal(mean, cov, 10000)
    Y = linear_model_2.evaluate(X)
    options = {
        "maxorder": 2,
        "maxiter": 100,
        "m": 2,
        "K": 1,
        "R": 10000,
        "alpha": 0.95,
        "lambdax": 0.01,
        "print_to_console": False,
    }
    Si = hdmr.analyze(problem, X, Y, **options)
    assert_allclose(
        Si["Sa"][0 : problem["num_vars"]],
        [0.28, 0.17, 0.10, 0.04, 0.02],
        atol=5e-2,
        rtol=1e-1,
    )
    assert_allclose(
        Si["Sb"][0 : problem["num_vars"]],
        [0.16, 0.16, 0.06, 0.00, 0.00],
        atol=5e-2,
        rtol=1e-1,
    )


def test_regression_rbd_fast():
    param_file = "src/SALib/test_functions/params/Ishigami.txt"
    problem = read_param_file(param_file)
    param_values = latin.sample(problem, 10000)

    Y = Ishigami.evaluate(param_values)

    Si = rbd_fast.analyze(problem, param_values, Y, print_to_console=False)
    assert_allclose(Si["S1"], [0.31, 0.44, 0.00], atol=5e-2, rtol=1e-1)


def test_regression_dgsm():
    param_file = "src/SALib/test_functions/params/Ishigami.txt"
    problem = read_param_file(param_file)
    param_values = finite_diff.sample(problem, 10000, delta=0.001)

    Y = Ishigami.evaluate(param_values)

    Si = dgsm.analyze(problem, param_values, Y, conf_level=0.95, print_to_console=False)

    assert_allclose(Si["dgsm"], [2.229, 7.066, 3.180], atol=5e-2, rtol=1e-1)


def test_regression_delta():
    param_file = "src/SALib/test_functions/params/Ishigami.txt"
    problem = read_param_file(param_file)
    param_values = latin.sample(problem, 10000)

    Y = Ishigami.evaluate(param_values)

    Si = delta.analyze(
        problem,
        param_values,
        Y,
        num_resamples=10,
        conf_level=0.95,
        print_to_console=True,
    )

    assert_allclose(Si["delta"], [0.210, 0.358, 0.155], atol=5e-2, rtol=1e-1)
    assert_allclose(Si["S1"], [0.31, 0.44, 0.00], atol=5e-2, rtol=1e-1)


def test_regression_delta_svm():
    xy_input_fn = "tests/data/delta_svm_regression_data.csv"
    inputs = np.loadtxt(xy_input_fn, delimiter=",", skiprows=1)

    X = inputs[:, 0]
    Y = inputs[:, 1]

    Ygrid = [
        0.0,
        0.3030303,
        0.60606061,
        0.90909091,
        1.21212121,
        1.51515152,
        1.81818182,
        2.12121212,
        2.42424242,
        2.72727273,
        3.03030303,
        3.33333333,
        3.63636364,
        3.93939394,
        4.24242424,
        4.54545455,
        4.84848485,
        5.15151515,
        5.45454545,
        5.75757576,
        6.06060606,
        6.36363636,
        6.66666667,
        6.96969697,
        7.27272727,
        7.57575758,
        7.87878788,
        8.18181818,
        8.48484848,
        8.78787879,
        9.09090909,
        9.39393939,
        9.6969697,
        10.0,
        10.3030303,
        10.60606061,
        10.90909091,
        11.21212121,
        11.51515152,
        11.81818182,
        12.12121212,
        12.42424242,
        12.72727273,
        13.03030303,
        13.33333333,
        13.63636364,
        13.93939394,
        14.24242424,
        14.54545455,
        14.84848485,
        15.15151515,
        15.45454545,
        15.75757576,
        16.06060606,
        16.36363636,
        16.66666667,
        16.96969697,
        17.27272727,
        17.57575758,
        17.87878788,
        18.18181818,
        18.48484848,
        18.78787879,
        19.09090909,
        19.39393939,
        19.6969697,
        20.0,
        20.3030303,
        20.60606061,
        20.90909091,
        21.21212121,
        21.51515152,
        21.81818182,
        22.12121212,
        22.42424242,
        22.72727273,
        23.03030303,
        23.33333333,
        23.63636364,
        23.93939394,
        24.24242424,
        24.54545455,
        24.84848485,
        25.15151515,
        25.45454545,
        25.75757576,
        26.06060606,
        26.36363636,
        26.66666667,
        26.96969697,
        27.27272727,
        27.57575758,
        27.87878788,
        28.18181818,
        28.48484848,
        28.78787879,
        29.09090909,
        29.39393939,
        29.6969697,
        30.0,
    ]

    m = [
        0.0,
        717.82142857,
        1435.64285714,
        2153.46428571,
        2871.28571429,
        3589.10714286,
        4306.92857143,
        5024.75,
        5742.57142857,
        6460.39285714,
        7178.21428571,
        7896.03571429,
        8613.85714286,
        9331.67857143,
        10049.5,
        10767.32142857,
        11485.14285714,
        12202.96428571,
        12920.78571429,
        13638.60714286,
        14356.42857143,
        15074.25,
        15792.07142857,
        16509.89285714,
        17227.71428571,
        17945.53571429,
        18663.35714286,
        19381.17857143,
        20099.0,
    ]

    num_resamples = 200
    conf_level = 0.95

    test_res = delta.bias_reduced_delta(
        Y, Ygrid, X, m, num_resamples, conf_level, Y.size
    )

    np.testing.assert_allclose(
        test_res, (0.6335098491949687, 0.026640611898969522), atol=0.005
    )
