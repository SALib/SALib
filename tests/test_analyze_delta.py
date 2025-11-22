from SALib.analyze import delta
from SALib.util import handle_seed
import numpy as np
import pandas as pd
import pytest


def create_base_dataframe(seed=42):
    rng = handle_seed(seed)

    n = 15000
    col_a = np.linspace(10, 80, n, endpoint=False) + rng.uniform(0, 70 / n, size=n)
    col_b = np.linspace(-20, 50, n, endpoint=False) + rng.uniform(0, 70 / n, size=n)
    rng.shuffle(col_a)
    rng.shuffle(col_b)
    df = pd.DataFrame(
        data={
            "output_col": np.clip(
                rng.normal(loc=22, scale=1.3, size=n), 17, 32
            ),  # Normal distribution centered at 22 with small std dev
            "column_a": col_a,
            "column_b": col_b,
        }
    )
    return df


def create_problemspec(df):
    Y = np.asarray(df["output_col"].values)
    cols_input = [col for col in df.columns if col != "output_col"]
    X = df[cols_input].values
    problem = {
        "num_vars": len(cols_input),
        "names": cols_input,
        "bounds": [[X[:, i].min(), X[:, i].max()] for i in range(len(cols_input))],
    }
    return problem, X, Y


def test_constant_input_column():
    """Show how delta handles constant values in an input feature column."""
    df = create_base_dataframe()
    df["column_consts"] = 7.77
    problem, X, Y = create_problemspec(df)
    with pytest.warns(UserWarning, match=r"has constant values"):
        _ = delta.analyze(problem=problem, X=X, Y=Y, num_resamples=3)


def test_binary_unbalanced():
    """If dataset too small for binning minimum"""
    df = create_base_dataframe()
    rng = handle_seed(42)

    n = len(df)
    n_zeroes = max(1, int(n * 0.001))
    zeroes_and_ones = np.array([0] * n_zeroes + [1] * (len(df) - n_zeroes))
    rng.shuffle(zeroes_and_ones)
    df["binary_small_zeroes"] = zeroes_and_ones
    problem, X, Y = create_problemspec(df)

    with pytest.warns(UserWarning, match=r"Only one bin remains"):
        _ = delta.analyze(problem=problem, X=X, Y=Y, num_resamples=3)


def test_binning_adaptation():
    """Data bin modification and adaptation."""
    df = create_base_dataframe()
    rng = handle_seed(42)
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
    rng.shuffle(value_col)
    df["binning_combination"] = value_col
    problem, X, Y = create_problemspec(df)

    with pytest.warns(UserWarning, match=r"Bin Merge Notice: Final no. bins is 6."):
        _ = delta.analyze(problem=problem, X=X, Y=Y, num_resamples=3)


def test_usererror_binspecs_datatype():
    """
    Show how delta handles when bininfo is specified as invalid datatype.
    If bin is specified incorrectly, it defaults to the default number of
    bins, equally distributed across X range. (default 10)
    """
    df = create_base_dataframe()
    rng = handle_seed(42)

    n = len(df)
    df["invalid_dtype"] = rng.integers(35, 500, n)

    bins_specs = {
        "invalid_dtype": "25",
    }

    problem, X, Y = create_problemspec(df)

    with pytest.warns(UserWarning, match=r"Input Error: Invalid custom bin dtype"):
        _ = delta.analyze(
            problem=problem, X=X, Y=Y, bins_specs=bins_specs, num_resamples=3
        )


def test_usererror_binspecs_notascending():
    """
    Show how delta handles when bininfo is specified as not ascending list.
    If bin specified is incorrect, then it defaults to the default number
    of bins equally distributed across X range. (default 10)
    """
    df = create_base_dataframe()
    rng = handle_seed(42)
    n = len(df)
    df["order_notascending"] = rng.integers(35, 500, n)
    bins_specs = {
        "order_notascending": [100, 300, 200, 450],
    }
    problem, X, Y = create_problemspec(df)
    with pytest.warns(UserWarning, match=r"Input Error: Bin edge error"):
        _ = delta.analyze(
            problem=problem, X=X, Y=Y, bins_specs=bins_specs, num_resamples=3
        )


def test_usererror_binspecs_outofrange():
    """
    Show how delta handles when bininfo is specified as list
    with boundaries out of X range.
    If bin specified is incorrect, then it defaults to the default
    number of bins equally distributed across X range. (default 10)
    """
    df = create_base_dataframe()
    rng = handle_seed(42)
    n = len(df)
    df["boundaries_outofXrange"] = rng.integers(35, 500, n)
    bins_specs = {
        "boundaries_outofXrange": [25, 100, 200, 300, 450],
    }
    problem, X, Y = create_problemspec(df)
    with pytest.warns(UserWarning, match=r"Input Error: Bin boundary error"):
        _ = delta.analyze(
            problem=problem, X=X, Y=Y, bins_specs=bins_specs, num_resamples=3
        )


def test_usererror_binspecs_nonnumeric():
    """
    Show how delta handles when bininfo is specified as list
    including non-numeric values.
    If bin specified is incorrect, then it defaults to the default
    number of bins equally distributed across X range. (default 10)
    """
    df = create_base_dataframe()
    rng = handle_seed(42)
    n = len(df)

    df["binvals_nonnumeric"] = rng.integers(35, 500, n)
    bins_specs = {
        "binvals_nonnumeric": [100, "200", 300, 450],
    }
    problem, X, Y = create_problemspec(df)

    with pytest.warns(UserWarning, match=r"Input Error: Bin value error"):
        _ = delta.analyze(
            problem=problem, X=X, Y=Y, bins_specs=bins_specs, num_resamples=3
        )


def test_binspecs_correct_use():
    """
    Show how delta handles when bininfo is specified correctly.
    Int, list (numeric, ascending), None, or unspecified.
    """
    df = create_base_dataframe()
    rng = handle_seed(42)
    n = len(df)

    # Examples of correct cases: integer, list of boundaries, None, or not specified in dict bins_specs
    df["correct_numbins"] = rng.integers(1, 550, n)
    df["correct_listbins"] = rng.integers(1, 550, n)
    df["correct_none"] = rng.integers(1, 550, n)
    df["correct_unspecified"] = rng.integers(1, 550, n)

    bins_specs = {
        "correct_numbins": 15,
        "correct_listbins": [100, 200, 300, 450],
        "correct_none": None,
    }
    problem, X, Y = create_problemspec(df)
    _ = delta.analyze(problem=problem, X=X, Y=Y, bins_specs=bins_specs, num_resamples=3)


def test_samplesize_mismatch():
    """If input X and output Y are not same number of rows."""
    df = create_base_dataframe()
    problem, X, Y = create_problemspec(df)
    Y = Y[::2]
    with pytest.raises(RuntimeError, match=r"Input X and output Y"):
        _ = delta.analyze(problem=problem, X=X, Y=Y, num_resamples=3)


def test_conf_level_too_low():
    """If conf level specified less than 0"""
    df = create_base_dataframe()
    problem, X, Y = create_problemspec(df)
    conf = -0.1
    with pytest.raises(RuntimeError, match=r"Confidence level"):
        _ = delta.analyze(problem=problem, X=X, Y=Y, conf_level=conf, num_resamples=3)


def test_conf_level_too_high():
    """If conf level specified greater than 1"""
    df = create_base_dataframe()
    problem, X, Y = create_problemspec(df)
    conf = 1.1
    with pytest.raises(RuntimeError, match=r"Confidence level"):
        _ = delta.analyze(problem=problem, X=X, Y=Y, conf_level=conf, num_resamples=3)


def test_invalid_userspec_method():
    """If user specified method is invalid (not delta, sobol, all)"""
    df = create_base_dataframe()
    problem, X, Y = create_problemspec(df)
    method = "something_else"

    with pytest.raises(ValueError, match=r"Method must be"):
        _ = delta.analyze(problem=problem, X=X, Y=Y, method=method, num_resamples=3)


def test_y_resamples_error():
    """If user specified y_resamples as greater than Y.size"""
    df = create_base_dataframe()
    rng = handle_seed(42)

    problem, X, Y = create_problemspec(df)
    y_resamples = rng.integers(len(Y) + 1, len(Y) + 500)

    with pytest.raises(RuntimeError, match=r"y_resamples must be"):
        _ = delta.analyze(
            problem=problem, X=X, Y=Y, y_resamples=y_resamples, num_resamples=3
        )


def test_invalid_columnnames_user():
    """
    If user provides any column name which has a forbidden
    suffix (raw, balanced, step, conf)
    """
    df = create_base_dataframe()
    rng = handle_seed(42)
    n = len(df)
    suffixes = ["_raw", "_balanced", "_step", "_conf"]
    suffix = suffixes[rng.integers(0, 4)]
    df[f"column{suffix}"] = rng.integers(35, 500, n)
    problem, X, Y = create_problemspec(df)

    with pytest.raises(ValueError, match=r"Forbidden column name"):
        _ = delta.analyze(problem=problem, X=X, Y=Y, num_resamples=3)


def test_data_bias_warning():
    """
    Warns user of potential data input bias,
    if raw and balanced deltas differ by > 0.1
    """
    df = create_base_dataframe()
    rng = handle_seed(42)
    n = len(df)
    output_col = df["output_col"].to_numpy()

    col_biased = np.zeros(n)
    col_biased[output_col < 20] = rng.uniform(0, 0.2, size=(output_col < 20).sum())
    col_biased[(output_col >= 20) & (output_col < 23)] = rng.uniform(
        0.8, 1.0, size=((output_col >= 20) & (output_col < 23)).sum()
    )
    col_biased[output_col >= 23] = rng.uniform(0.3, 0.5, size=(output_col >= 23).sum())

    df["col_biased"] = col_biased
    problem, X, Y = create_problemspec(df)

    with pytest.warns(UserWarning, match=r"Potential Bias Notice:"):
        _ = delta.analyze(
            problem=problem, X=X, Y=Y, num_resamples=3, bins_specs={"col_biased": 2}
        )


def test_dataset_small_size():
    """Warns user if dataset size too small for number of bins"""
    df = create_base_dataframe()
    n = len(df)
    problem, X, Y = create_problemspec(df)

    with pytest.warns(UserWarning, match=r"Dataset size error:"):
        _ = delta.analyze(
            problem=problem,
            X=X,
            Y=Y,
            num_resamples=3,
            y_resamples=max(1, int(0.01 * n)),
        )
