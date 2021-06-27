import numpy as np
import pandas as pd
from SALib.sample import saltelli, morris as morris_sample, \
                         finite_diff, fast_sampler, ff as ff_sample, latin
from SALib.analyze import sobol, morris, dgsm, fast, ff, rbd_fast
from SALib.test_functions import Ishigami, Sobol_G


def test_morris_to_df():
    params = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8']

    problem = {
        'num_vars': 8,
        'names': params,
        'groups': None,
        'bounds': [[0.0, 1.0],
                   [0.0, 1.0],
                   [0.0, 1.0],
                   [0.0, 1.0],
                   [0.0, 1.0],
                   [0.0, 1.0],
                   [0.0, 1.0],
                   [0.0, 1.0]]
    }

    param_values = morris_sample.sample(problem, N=1000, num_levels=4,
                                        optimal_trajectories=None)
    Y = Sobol_G.evaluate(param_values)
    Si = morris.analyze(problem, param_values, Y)
    Si_df = Si.to_df()

    assert isinstance(Si_df, pd.DataFrame), \
        "Morris Si: Expected DataFrame, got {}".format(type(Si_df))

    expected_index = set(params)
    assert set(Si_df.index) == expected_index, "Incorrect index in DataFrame"

    col_names = ['mu', 'mu_star', 'sigma', 'mu_star_conf']
    assert set(Si_df.columns) == set(col_names), \
        "Unexpected column names in DataFrame. Expected {}, got {}".format(
            col_names, Si_df.columns)
# End test_morris_to_df()


def test_sobol_to_df():
    params = ['x1', 'x2', 'x3']
    problem = {
        'num_vars': 3,
        'names': params,
        'bounds': [[-np.pi, np.pi]]*3
    }

    X = saltelli.sample(problem, 512)
    Y = Ishigami.evaluate(X)
    Si = sobol.analyze(problem, Y, print_to_console=False)
    total, first, second = Si.to_df()

    assert isinstance(total, pd.DataFrame), \
        "Total Si: Expected DataFrame, got {}".format(type(total))
    assert isinstance(first, pd.DataFrame), \
        "First Si: Expected DataFrame, got {}".format(type(first))
    assert isinstance(second, pd.DataFrame), \
        "Second Si: Expected DataFrame, got {}".format(type(second))

    expected_index = set(params)
    assert set(total.index) == expected_index, \
        "Index for Total Si are incorrect"
    assert set(first.index) == expected_index, \
        "Index for first order Si are incorrect"
    assert set(second.index) == set([('x1', 'x2'),
                                     ('x1', 'x3'),
                                     ('x2', 'x3')]), \
        "Index for second order Si are incorrect"
# End test_sobol_to_df()


def test_dgsm_to_df():
    params = ['x1', 'x2', 'x3']
    problem = {
        'num_vars': 3,
        'names': params,
        'groups': None,
        'bounds': [[-3.14159265359, 3.14159265359],
                   [-3.14159265359, 3.14159265359],
                   [-3.14159265359, 3.14159265359]]
    }

    param_values = finite_diff.sample(problem, 1000, delta=0.001)
    Y = Ishigami.evaluate(param_values)

    Si = dgsm.analyze(problem, param_values, Y, print_to_console=False)
    Si_df = Si.to_df()

    assert isinstance(Si_df, pd.DataFrame), \
        "DGSM Si: Expected DataFrame, got {}".format(type(Si_df))
    assert set(Si_df.index) == set(params), "Incorrect index in DataFrame"

    col_names = ['vi', 'vi_std', 'dgsm', 'dgsm_conf']
    assert set(Si_df.columns) == set(col_names), \
        "Unexpected column names in DataFrame"
# End test_dgsm_to_df()


def test_fast_to_df():
    params = ['x1', 'x2', 'x3']
    problem = {
        'num_vars': 3,
        'names': params,
        'groups': None,
        'bounds': [[-3.14159265359, 3.14159265359],
                   [-3.14159265359, 3.14159265359],
                   [-3.14159265359, 3.14159265359]]
    }

    param_values = fast_sampler.sample(problem, 1000)
    Y = Ishigami.evaluate(param_values)

    Si = fast.analyze(problem, Y, print_to_console=False)
    Si_df = Si.to_df()

    expected_index = set(params)
    assert isinstance(Si_df, pd.DataFrame), \
        "FAST Si: Expected DataFrame, got {}".format(type(Si_df))
    assert set(Si_df.index) == expected_index, "Incorrect index in DataFrame"

    col_names = set(['S1', 'ST', 'S1_conf', 'ST_conf'])
    assert set(Si_df.columns) == col_names, \
        "Unexpected column names in DataFrame. Expected {}, got {}".format(
            col_names, Si_df.columns)
# End test_fast_to_df()


def test_ff_to_df():
    params = ['x1', 'x2', 'x3']
    main_index = params + ['dummy_0']

    problem = {
        'num_vars': 3,
        'names': params,
        'groups': None,
        'bounds': [[-3.14159265359, 3.14159265359],
                   [-3.14159265359, 3.14159265359],
                   [-3.14159265359, 3.14159265359]]
    }

    X = ff_sample.sample(problem)
    Y = X[:, 0] + (0.1 * X[:, 1]) + ((1.2 * X[:, 2]) * (0.2 + X[:, 0]))
    Si = ff.analyze(problem, X, Y, second_order=True, print_to_console=False)
    main_effect, inter_effect = Si.to_df()

    assert isinstance(main_effect, pd.DataFrame), \
        "FF ME: Expected DataFrame, got {}".format(type(main_effect))
    assert isinstance(main_effect, pd.DataFrame), \
        "FF IE: Expected DataFrame, got {}".format(type(inter_effect))
    assert set(main_effect.index) == set(main_index), \
        "Incorrect index in Main Effect DataFrame"

    inter_index = set([('x1', 'x2'),
                       ('x1', 'x3'),
                       ('x2', 'x3'),
                       ('x1', 'dummy_0'),
                       ('x2', 'dummy_0'),
                       ('x3', 'dummy_0')])
    assert set(inter_effect.index) == inter_index, \
        "Incorrect index in Interaction Effect DataFrame"
# End test_ff_to_df()


def test_rbd_to_df():
    params = ['x1', 'x2', 'x3']
    problem = {
        'num_vars': 3,
        'names': params,
        'groups': None,
        'bounds': [[-3.14159265359, 3.14159265359],
                   [-3.14159265359, 3.14159265359],
                   [-3.14159265359, 3.14159265359]]
    }

    param_values = latin.sample(problem, 1000)
    Y = Ishigami.evaluate(param_values)
    Si = rbd_fast.analyze(problem, param_values, Y, print_to_console=False)
    Si_df = Si.to_df()

    assert isinstance(Si_df, pd.DataFrame), \
        "RBD Si: Expected DataFrame, got {}".format(type(Si_df))
    assert set(Si_df.index) == set(params), "Incorrect index in DataFrame"

    col_names = set(['S1', 'S1_conf'])
    assert set(Si_df.columns) == col_names, \
        "Unexpected column names in DataFrame. Expected {}, got {}".format(
            col_names, Si_df.columns)
# End test_rbd_to_df()
