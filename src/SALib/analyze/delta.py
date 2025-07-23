from typing import Optional, Dict
from scipy.stats import norm, gaussian_kde, rankdata

import numpy as np
import pandas as pd

from . import common_args
from ..util import read_param_file, ResultDict

import warnings
import json


class AnalysisError(Exception):
    def __init__(self, cmd, note=None):
        self.cmd = cmd
        self.note = note or cmd
        super().__init__(self.note)
        warnings.warn(self.cmd)


class BootstrapError(AnalysisError):
    pass


class SampleSizeError(AnalysisError):
    pass


def exception_handler(e, name, Ysize):
    if isinstance(e, AnalysisError):
        return e.note
    elif isinstance(e, np.linalg.LinAlgError):
        raise SampleSizeError(
            f"ERROR [{name}]: Singular matrix - likely due to degenerate input or small sample size (N={Ysize})"
        ) from e
    else:
        warnings.warn(f"Unhandled exception: {type(e).__name__} - {e}")
        return str(e)


def custom_warning_formatter(message, category, filename, lineno, line=None):
    return f"{message}\n"


def analyze(
    problem: Dict,
    X: np.ndarray,
    Y: np.ndarray,
    num_resamples: int = 100,
    conf_level: float = 0.95,
    print_to_console: bool = False,
    seed: Optional[int] = None,
    y_resamples: Optional[int] = None,
    method: str = "all",
    bootstrap_savedf: Optional[str] = None,
    bins_specs: Dict = {},
) -> Dict:
    """Perform Delta Moment-Independent Analysis on model outputs.

    Returns a dictionary with keys 'delta_balanced', 'delta_balanced_conf', 'delta_raw', 'delta_raw_conf', 'delta_step', 'delta_step_conf',
    'S1', 'S1_conf', 'names', where each entry is a list of size D (the number of parameters) containing
    the indices in the same order as the input file and 'names'

    The different delta configurations/methods 'balanced', 'step', 'raw', their corresponding confidence scores (*_conf), and other keys in returned dictionary:
        - 'names'
            input column/feature names in X

        - 'notes'
            Notes and insights on analysis, warnings, errors, etc.

        - 'balanced'
            Divides the input into bins during bootstrapping (default 10) and divides bootstrap subset equally among those bins

        - 'step'
            Generates all entries in the input into 1 (X>0) or 0 (X<=0) and bootstrap subset is approximately 50%

        - 'raw'
            Bootstraps the input column as is, with random bootstrap sampling with replacement

        - 's1'
            Sobol' first indices, on raw dataset with no bootstrap manipulations (random, with replacement)


    Notes
    -----
    Compatible with:
        all samplers

    Interpretaion:
        'balanced' tells us how important variation in that input feature is on the output feature.
            - Answers the question: "How important is input feature X on output Y?"

        'step' tells us how important that input feature as active (versus inactive) is on the output feature.
            - Answers the question: "What is the impact of input feature X when it is greater than zero/active,
                versus inactive, regardless of its actual value?"

        'raw' tells us how much the input feature affects variation in the output feature, specifically in this
            dataset and its composition. This score may be skewed or biased toward frequently occurring values
            in the dataset. (eg. if most values of the input feature lie between 15-20 in your dataset, then the
            delta score will likely primarily be biased toward influence within that range, rather than its total
            influence over its entire input range)
            - Answers the question: "How important is input feature X, and how prominent is its influence specifically
                in my dataset?"

        'notes' contains flags, warnings, or error messages.
            - Will raise a flag if 'balanced' and 'raw' delta scores differ by >0.1. This suggests that the input dataset is
                likely skewed or biased with frequently occuring values in a specific range.
            - Raises errors if binning cannot be executed, if there aren't enough/any zeroes/>0s present to do step analysis, etc.
            - Guides user with results interpretation


    Examples
    --------
        >>> X = latin.sample(problem, 1000)
        >>> Y = Ishigami.evaluate(X)
        >>> Si = delta.analyze(problem, X, Y, print_to_console=True)


    Parameters
    ----------
    problem : dict
        The problem definition
    X: numpy.matrix
        A NumPy matrix containing the model inputs
    Y : numpy.array
        A NumPy array containing the model outputs
    num_resamples : int
        The number of resamples when computing confidence intervals (default 100)
    conf_level : float
        The confidence interval level (default 0.95)
    print_to_console : bool
        Print results directly to console (default False)
    y_resamples : int, optional
        Number of samples to use when resampling (bootstrap) (default None)
    method : {"all", "delta", "sobol"}, optional
        Whether to compute "delta", "sobol" or both ("all") indices (default "all")
    bootstrap_savedf : str, optional
        User inputs path or filename if they want to save a bootstrap sample to inspect subset composition
    bins_specs : dict, optional
        Dict with parameter title as key and bin_input as entry.
            If bin_input is an int, number of bins.
            If bin_input is a list, it specifies bin boundaries ( number of bins = len(list)+1 )
            Default is 10 bins equally distributed across input feature range


    References
    ----------
    1. Borgonovo, E. (2007). "A new uncertainty importance measure."
           Reliability Engineering & System Safety, 92(6):771-784,
           doi:10.1016/j.ress.2006.04.015.

    2. Plischke, E., E. Borgonovo, and C. L. Smith (2013). "Global
           sensitivity measures from given data." European Journal of
           Operational Research, 226(3):536-550, doi:10.1016/j.ejor.2012.11.047.
    """
    if seed:
        np.random.seed(seed)

    valid_methods = {"delta", "sobol", "all"}
    warnings.formatwarning = custom_warning_formatter
    warnings.simplefilter("once")

    if X.shape[0] != Y.shape[0]:
        raise RuntimeError("Input X and output Y must have the same number of rows")
    if not 0 < conf_level < 1:
        raise RuntimeError("Confidence level must be between 0-1.")
    if method not in valid_methods:
        raise ValueError(f"Method must be one of {valid_methods}")

    delta_warn_threshold = 0.1
    D = problem["num_vars"]
    if y_resamples is None:
        y_resamples = Y.size
    methods = ["delta", "sobol"] if method == "all" else [method]
    if bootstrap_savedf is not None:
        if not bootstrap_savedf.endswith(".parquet"):
            bootstrap_savedf = bootstrap_savedf + ".parquet"

    if not y_resamples <= Y.size:
        raise RuntimeError(
            "y_resamples must be less than or equal to the total number of samples"
        )

    # equal frequency partition
    exp = 2.0 / (7.0 + np.tanh((1500.0 - y_resamples) / 500.0))
    M = int(np.round(min(int(np.ceil(y_resamples**exp)), 48)))
    m = np.linspace(0, y_resamples, M + 1)
    Ygrid = np.linspace(np.min(Y), np.max(Y), 100)

    # S
    keys = ["notes"]
    if "delta" in methods:
        keys += [
            "delta_balanced",
            "delta_balanced_conf",
            "delta_raw",
            "delta_raw_conf",
            "delta_step",
            "delta_step_conf",
        ]
    if "sobol" in methods:
        keys += ["S1", "S1_conf"]
    S = ResultDict((k, np.full(D, np.nan)) for k in keys)

    nameslist = problem["names"]
    if any(
        any(col.endswith(s) for s in ["_raw", "_balanced", "_step", "_conf"])
        for col in nameslist
    ):
        raise ValueError(
            "Forbidden column name suffix: columns may not end with ['_raw', '_balanced', '_step', '_conf'] "
        )

    S["names"] = problem["names"]
    notes = []
    bootstrap_matrix = {}

    for i in range(D):
        name = problem["names"][i]
        X_i = X[:, i]
        min_class_size = max(int(len(X_i) * 0.005), 10)
        bininfo = bins_specs.get(name, None)
        bin_edges, note = check_specified_bininfo(
            bininfo=bininfo, Xmin=X_i.min(), Xmax=X_i.max(), paramname=name
        )
        try:
            if "delta" in methods:
                for mode in ["balanced", "raw", "step"]:
                    if mode == "step":
                        zeroes = np.sum(X_i <= 0)
                        nonzeroes = np.sum(X_i > 0)
                        if zeroes < min_class_size or nonzeroes < min_class_size:
                            S["delta_step"][i] = np.nan
                            S["delta_step_conf"][i] = np.nan
                            note += f"Step: too few samples. {zeroes} (<=0),   {nonzeroes} (>0); "
                            continue

                    S[f"delta_{mode}"][i], S[f"delta_{mode}_conf"][i], boot_idx = (
                        bias_reduced_delta(
                            Y,
                            Ygrid,
                            X_i,
                            m,
                            num_resamples,
                            conf_level,
                            y_resamples,
                            mode,
                            bin_edges,
                            name,
                            min_class_size,
                        )
                    )
                    if bootstrap_savedf is not None:
                        bootstrap_matrix[f"{name}_{mode}"] = (
                            X_i[boot_idx]
                            if len(boot_idx) == y_resamples
                            else np.full(y_resamples, np.nan)
                        )
                diff = abs(S["delta_balanced"][i] - S["delta_raw"][i])
                if diff > delta_warn_threshold:
                    note += f"Raw delta differs from balanced by {diff:.2f}; "
                    warnings.warn(
                        f"[{name}] Potential Bias Notice: Raw delta score differs from balanced delta by {diff:.2f}. Potential dataset bias, take care in interpretation."
                    )

            if "sobol" in methods:
                ind = np.random.randint(Y.size, size=y_resamples)
                S["S1"][i] = sobol_first(Y[ind], X_i[ind], m)
                S["S1_conf"][i] = sobol_first_conf(
                    Y, X_i, m, num_resamples, conf_level, y_resamples
                )
            notes.append(note.strip() if note else "")
        except Exception as e:
            note = exception_handler(e, name, Y.size)
            notes.append(note)
            continue
        S["notes"] = notes
    if print_to_console:
        summary_dict = {"names": problem["names"], "notes": notes}
        if "delta_balanced" in S:
            summary_dict.update(
                {
                    "delta_balanced": S["delta_balanced"],
                    "delta_step": S["delta_step"],
                    "delta_raw": S["delta_raw"],
                    "delta_balanced_conf": S["delta_balanced_conf"],
                    "delta_step_conf": S["delta_step_conf"],
                    "delta_raw_conf": S["delta_raw_conf"],
                }
            )
        if "S1" in S:
            summary_dict.update(
                {
                    "s1": S["S1"],
                    "s1_conf": S["S1_conf"],
                }
            )
        df_summary = pd.DataFrame(summary_dict)
        print(df_summary.to_string(index=False))
    if bootstrap_savedf is not None:
        df_boot = pd.DataFrame(bootstrap_matrix)
        try:
            df_boot.to_parquet(bootstrap_savedf, engine="pyarrow", compression="snappy")
        except Exception as e:
            warnings.warn(
                f"WARNING: Failed to write bootstrap file to {bootstrap_savedf}. Reason: {e}"
            )
    return S


def check_specified_bininfo(bininfo, Xmin, Xmax, paramname):
    """Validate user-specified bin edges or number of bins; fallback to default if invalid."""
    defaultbins = 10  # 5% * (1/0.005)
    if bininfo is None:
        return np.linspace(Xmin, Xmax, defaultbins + 1), ""
    elif isinstance(bininfo, int):
        return np.linspace(Xmin, Xmax, bininfo + 1), ""
    elif isinstance(bininfo, (list, np.ndarray)):
        bin_edges = np.array(bininfo)
        if not np.issubdtype(bin_edges.dtype, np.number):
            warnings.warn(
                f"[{paramname}] USER INPUT ERROR: Bin value error: Custom bin list only contain numeric values. Resorting back to default binning ({defaultbins} bins)."
            )
            return (
                np.linspace(Xmin, Xmax, defaultbins + 1),
                f"Custom bin list had non-numeric values. Used default {defaultbins} bins; ",
            )
        elif np.any(bin_edges < Xmin) or np.any(bin_edges > Xmax):
            warnings.warn(
                f"[{paramname}] USER INPUT ERROR: Bin boundary error: Custom bin boundaries must be within the X range. Resorting back to default binning ({defaultbins} bins)."
            )
            return (
                np.linspace(Xmin, Xmax, defaultbins + 1),
                f"Custom bin bounds out of X range. Used default {defaultbins} bins; ",
            )
        elif not np.all(np.diff(bin_edges) > 0):
            warnings.warn(
                f"[{paramname}] USER INPUT ERROR: Bin edge error: Custom bin edges must be ascending. Resorting back to default binning ({defaultbins} bins)."
            )
            return (
                np.linspace(Xmin, Xmax, defaultbins + 1),
                f"Custom bin edges not ascending. Used default {defaultbins} bins; ",
            )
        return np.concatenate(([Xmin - 1e-9], bin_edges, [Xmax + 1e-9])), ""
    else:
        warnings.warn(
            f"[{paramname}] USER INPUT ERROR: Invalid custom bin dtype: Must be int, list or None. Resorting back to default binning ({defaultbins} bins)."
        )
        return (
            np.linspace(Xmin, Xmax, defaultbins + 1),
            f"Custom bin invalid dtype. Used default {defaultbins} bins; ",
        )


def calc_delta(Y, Ygrid, X, m):
    """Plischke et al. (2013) delta index estimator (eqn 26) for d_hat."""
    N = len(Y)

    fy = gaussian_kde(Y, bw_method="silverman")(Ygrid)
    xr = rankdata(X, method="ordinal")

    d_hat = 0.0
    total_bins = len(m) - 1
    num_skipped = 0
    for j in range(total_bins):
        ix = np.where((xr > m[j]) & (xr <= m[j + 1]))[0]
        nm = len(ix)
        if nm == 0:
            continue  # Skip empty bins. Even if it doesn't throw an error, can't estimate local distribution for gaussian_kde from empty data

        Y_ix = Y[ix]
        if np.ptp(Y_ix) != 0.0:
            fyc = gaussian_kde(Y_ix, bw_method="silverman")(Ygrid)
            fy_ = np.abs(fy - fyc)
        else:
            num_skipped += 1
            continue

        d_hat += (nm / (2 * N)) * np.trapezoid(fy_, Ygrid)
    if num_skipped / total_bins > 0.35:
        warnings.warn(
            f"{num_skipped}/{total_bins} bins skipped (>35%)"
            "This may indicate under-sampling or poor binning."
            "Try increasing sample size (y_resamples) or decreasing the number of bins (M)."
        )
    return d_hat


def bias_reduced_delta(
    Y,
    Ygrid,
    X,
    m,
    num_resamples,
    conf_level,
    y_resamples,
    mode,
    bin_edges,
    paramname,
    min_class_size,
):
    """Plischke et al. 2013 bias reduction technique (eqn 30)"""
    d = np.empty(num_resamples)

    ind = obtain_resampled_subset(
        X, y_resamples, mode, bin_edges, paramname, min_class_size, warn_print=True
    )
    d_hat = calc_delta(Y[ind], Ygrid, X[ind], m)

    try:
        r = [obtain_resampled_subset(X, y_resamples, mode, bin_edges, paramname, min_class_size, warn_print=False) for _ in range(num_resamples)]
    except ValueError as e:
        raise RuntimeError(f"BOOTSTRAP ERROR: [{paramname}][{mode}]: {e}") from e

    for i, r_i in enumerate(r):
        d[i] = calc_delta(Y[r_i], Ygrid, X[r_i], m)

    d = 2.0 * d_hat - d
    return (d.mean(), norm.ppf(0.5 + conf_level / 2) * d.std(ddof=1), ind)


def obtain_bins_indices(bin_edges, X, min_class_size):
    """
    Returns indices for all specified bins, ensuring 
    that all bins are at least minimum_class_size
    """
    init_indices = []
    for i in range(len(bin_edges) - 1):
        if i == 0:
            idx = np.where((X >= bin_edges[i]) & (X <= bin_edges[i + 1]))[0]
        else:
            idx = np.where((X > bin_edges[i]) & (X <= bin_edges[i + 1]))[0]
        init_indices.append(idx)
    final_indices = []  # for the case where some bins are underpopulated
    i = 0
    while i < len(init_indices):
        current_bin = init_indices[i]
        while len(current_bin) < min_class_size and i + 1 < len(init_indices):
            current_bin = np.concatenate((current_bin, init_indices[i + 1]))
            i += 1
        final_indices.append(current_bin)
        i += 1
    return init_indices, final_indices


def obtain_resampled_subset(
    X, y_resamples, mode, bin_edges, paramname, min_class_size, warn_print=False
):
    """Returns subset indices for X"""
    N = len(X)
    X = np.asarray(X)
    if X.max() == X.min():
        raise BootstrapError(
            f"[{paramname}] has constant values, cannot bin or calculate delta.",
            "Cannot bin or calc delta from constants; ",
        )
    if mode == "raw":
        return np.random.randint(N, size=y_resamples)

    elif mode in ["balanced", "step"]:
        indices = []
        X_select = X
        binselectionsize = y_resamples
        if mode == "step":
            zeroes = np.where(X <= 0)[0]
            nonzeroes = np.where(X > 0)[0]

            n_zeroes = int(0.5 * y_resamples)
            binselectionsize = y_resamples - n_zeroes
            min_class_size = min_class_size // 2

            chosen_zeroes = np.random.choice(zeroes, size=n_zeroes, replace=True)
            indices.append(chosen_zeroes)
            X_select = X[nonzeroes]
            bin_edges, _ = check_specified_bininfo(
                bininfo=5, Xmin=X_select.min(), Xmax=X_select.max(), paramname=paramname
            )

        init_indices, final_indices = obtain_bins_indices(
            bin_edges, X_select, min_class_size
        )
        n_bins = len(final_indices)
        n_per_bin = binselectionsize // n_bins
        remaining = binselectionsize - (n_per_bin * n_bins)

        if n_bins < 2:
            raise AnalysisError(
                f"[{paramname}][{mode}] Only one bin remains. Revisit whether processing method is suitable for this parameter, or increase dataset size or spread. Highly biased input.",
                f"[{mode}] Single valid bin. Insufficient spread in feature, highly biased input; ",
            )
        if n_bins != len(init_indices):
            if warn_print:
                warnings.warn(
                    f"[{paramname}][{mode}] Bin Merge Notice: Final no. bins is {len(final_indices)}. Min samples per bin: {min_class_size}"
                )
        if n_per_bin < min_class_size:
            raise SampleSizeError(
                f"[{paramname}][{mode}] Dataset size error: Dataset is not large enough for number of bins.",
                f"[{mode}]: Dataset too small for no. bins; ",
            )

        for i, bin_idx in enumerate(final_indices):
            if len(bin_idx) == 0:
                continue
            binsize = n_per_bin + (1 if i < remaining else 0)
            sampled = np.random.choice(bin_idx, size=binsize, replace=True)
            if mode == "step":
                sampled = nonzeroes[sampled]  # type: ignore
            indices.append(sampled)
        return np.concatenate(indices)
    else:
        raise ValueError(
            f"{paramname} Issue with mode {mode} suffix. \
                This line should never execute."
        )


def sobol_first(Y, X, m):
    # pre-process to catch constant array
    # see: https://github.com/numpy/numpy/issues/9631
    if np.ptp(Y) == 0.0:
        # Catch constant results
        # If Y does not change then it is not sensitive to anything...
        return 0.0

    xr = rankdata(X, method="ordinal")
    Vi = 0
    N = len(Y)
    Y_mean = Y.mean()
    for j in range(len(m) - 1):
        ix = np.where((xr > m[j]) & (xr <= m[j + 1]))[0]
        nm = len(ix)
        Vi += (nm / N) * ((Y[ix].mean() - Y_mean) ** 2)

    return Vi / np.var(Y)


def sobol_first_conf(Y, X, m, num_resamples, conf_level, y_resamples):
    s = np.zeros(num_resamples)

    N = len(Y)
    r = np.random.choice(
        N, size=(num_resamples, y_resamples), replace=True
    )  # bootstrap-like behaviour, on raw vals without removing input bias

    for i in range(num_resamples):
        r_i = r[i, :]
        s[i] = sobol_first(Y[r_i], X[r_i], m)

    return norm.ppf(0.5 + conf_level / 2) * s.std(ddof=1)


def cli_parse(parser):
    parser.add_argument(
        "-X",
        "--model-input-file",
        type=str,
        required=True,
        default=None,
        help="Model input file",
    )
    parser.add_argument(
        "-r",
        "--resamples",
        type=int,
        required=False,
        default=10,
        help="Number of bootstrap resamples for \
                           Sobol confidence intervals",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        required=False,
        default="all",
        help="Method to compute sensitivities \
                    'delta', 'sobol' or 'all'",
    )
    parser.add_argument(
        "--y_resamples",
        type=int,
        required=False,
        default=None,
        help="Number of samples to use when \
                    resampling (bootstrap)",
    )
    parser.add_argument(
        "--bootstrap_savedf",
        type=str,
        help="Optional path to save bootstrap \
                feature values",
    )
    parser.add_argument(
        "--bins_specs",
        type=str,
        default=None,
        help="Optional: JSON string or path to a \
                JSON file specifying bin info per \
                input feature",
    )

    return parser


def cli_action(args):
    problem = read_param_file(args.paramfile)
    Y = np.loadtxt(
        args.model_output_file, 
        delimiter=args.delimiter, 
        usecols=(args.column,)
    )
    X = np.loadtxt(args.model_input_file, delimiter=args.delimiter, ndmin=2)
    if len(X.shape) == 1:
        X = X.reshape((len(X), 1))

    try:
        if args.bins_specs is None:
            bins_specs = {}
        elif args.bins_specs.strip().startswith("{"):
            bins_specs = json.loads(args.bins_specs)
        else:
            with open(args.bins_specs) as f:
                bins_specs = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        raise ValueError(
            f"Failed to parse bins_specs. Must be a JSON string or \
                file path. Error: {e}"
        )

    analyze(
        problem,
        X,
        Y,
        num_resamples=args.resamples,
        print_to_console=True,
        seed=args.seed,
        method=args.method,
        y_resamples=args.y_resamples,
        bootstrap_savedf=args.bootstrap_savedf,
        bins_specs=bins_specs,
    )


if __name__ == "__main__":
    common_args.run_cli(cli_parse, cli_action)
