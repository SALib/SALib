from SALib.analyze import delta
import numpy as np
import pandas as pd

def create_base_dataframe(seed=None):
    if seed is None: seed=42
    np.random.seed(seed)
    
    n=15000
    c_nz = 0.1 + (0.8 - 0.1) * np.random.beta(a=5, b=1.5, size=int(0.25*n))
    c_col = np.zeros(n)
    c_col[:int(0.25*n)] = c_nz
    np.random.shuffle(c_col)
    df = pd.DataFrame(data={
        'output_col': np.clip(np.random.normal(loc=22, scale=1.3, size=n), 17, 32),     # Normal distribution centered at 22 with small std dev
        'column_a': np.clip(np.random.normal(loc=27.5, scale=2.0, size=n), 15, 40),     # Normal distribution centered at 27.5 with small std dev
        'column_b': np.clip(np.random.normal(loc=82.5, scale=5.0, size=n), 50, 100),    # Normal distribution centered at 82.5, constrained between 50-100
        'column_c': c_col,                                                              # 75% zeros, 25% values between 0.8 and 0.1 (more around 0.1)
    })
    return df

def create_problemspec(df):
    Y = np.asarray(df['output_col'].values)
    cols_input = [col for col in df.columns if col!='output_col']
    X = df[cols_input].values
    problem = {
    'num_vars': len(cols_input),
    'names': cols_input,
    'bounds': [[X[:, i].min(), X[:, i].max()] for i in range(len(cols_input))]
    }
    return problem, X, Y


def test_constant_input_column():
    """Show how delta handles constant values in an input feature column."""
    df = create_base_dataframe()
    df['column_consts'] = 7.77
    problem, X, Y = create_problemspec(df)
    Si = delta.analyze(problem=problem, X=X, Y=Y)
    return Si

def test_zeroes_and_ones():
    """Show how delta handles data if there are too few values for binning minimum. 20 zeroes and the rest are ones. Plus other splits with no errors"""
    df = create_base_dataframe()
    zeroes_and_ones = np.array([0]*20 + [1]*(len(df) - 20))
    np.random.shuffle(zeroes_and_ones)
    df['binary_small_zeroes'] = zeroes_and_ones

    df['random_binary'] = np.random.randint(0,2, len(df)) # more or less 50% zeroes and ones split, no error
    equal_part = np.array([1, 2, 3] * (len(df) // 3) + [1]*(len(df) % 3))
    np.random.shuffle(equal_part)
    df['equal_partition_123'] = equal_part # more or less equal between 1, 2, 3, no error

    problem, X, Y = create_problemspec(df)
    Si = delta.analyze(problem=problem, X=X, Y=Y)
    return Si

def test_binning_adaptation():
    """Show how delta handles data bin modification and adaptation."""
    df = create_base_dataframe()
    n = len(df)
    counts = {
        0: int(0.0004 * n),
        1: int(0.0005 * n),
        2: int(0.0010 * n),
        3: int(0.0100 * n),
        4: int(0.0200 * n),
        5: int(0.2500 * n),
        6: int(0.2500 * n),
        7: int(0.3700 * n),
        8: int(0.0020 * n),
        9: int(0.0009 * n),
    }
    total_assigned = sum(counts.values())
    counts[10] = n - total_assigned
    value_col = np.array([val for val, count in counts.items() for _ in range(count)])
    np.random.shuffle(value_col)
    df['binning_combination'] = value_col

    problem, X, Y = create_problemspec(df)
    Si = delta.analyze(problem=problem, X=X, Y=Y)
    return Si


def test_binning_binspecs():
    """
    Show how delta handles when bininfo is specified as invalid datatype, incorrect values, invalid order, or others.
    If bin specified is incorrect, then it defaults to the default number of bins equally distributed across X range. (default 10)
    """
    df = create_base_dataframe()
    n = len(df)
    df['invalid_dtype'] = np.random.randint(35, 500, n)
    df['order_notascending'] = np.random.randint(35, 500, n)
    df['boundaries_outofXrange'] = np.random.randint(35, 500, n)
    df['binvals_nonnumeric'] = np.random.randint(35, 500, n)
    
    # Examples where we illustrate correct cases: integer, list of boundaries, None, or unspecified in dict bins_specs
    df['correct_numbins'] = np.random.randint(1, 550, n)
    df['correct_listbins'] = np.random.randint(1, 550, n)
    df['correct_none'] = np.random.randint(1, 550, n)
    df['correct_unspecified'] = np.random.randint(1, 550, n)

    bins_specs = {
        'invalid_dtype': '25',
        'order_notascending': [100, 300, 200, 450],
        'boundaries_outofXrange': [25, 100, 200, 300, 450],
        'binvals_nonnumeric': [100, '200', 300, 450],
        'correct_numbins': 15,
        'correct_listbins': [100, 200, 300, 450],
        'correct_none': None,
    }

    problem, X, Y = create_problemspec(df)
    Si = delta.analyze(problem=problem, X=X, Y=Y, bins_specs=bins_specs)
    return Si