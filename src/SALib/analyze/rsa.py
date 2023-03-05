from typing import Dict
from types import MethodType
import warnings

import numpy as np
import pandas as pd
from scipy.stats import anderson_ksamp

from . import common_args
from ..util import read_param_file, ResultDict, extract_group_names, _check_groups


def analyze(
    problem: Dict,
    X: np.ndarray,
    Y: np.ndarray,
    bins: int = 20,
    target: str = "Y",
    print_to_console: bool = False,
    seed: int = None,
):
    """
    Perform Regional Sensitivity Analysis (RSA), also known as Monte Carlo Filtering.

    In a usual RSA, a desirable region of output space is defined. Outputs which fall within
    this region is categorized as being "behavioral" ($B$), and those outside are described
    as being "non-behavioral" ($\\bar{B}$). The input factors are also partitioned into
    behavioral and non-behavioral subsets, such that $f(X_{i}|B) -> (Y|B)$ and
    $f(X_{i}|\\bar{B}) -> (Y|\\bar{B})$. The distribution between the two sub-samples are
    compared for each factor. The greater the difference between the two distributions, the
    more important the given factor is in driving model outputs.

    The approach implemented in SALib partitions factor or output space into $b$ bins
    (default: 20) according to their percentile values. Output space is targeted for analysis
    by default (`target="Y"`), such that $(Y|b_{i})$ is mapped back to $X_{i}|b_{i}$.
    In other words, we treat outputs falling within a given bin ($b_{i}$) corresponding to
    their inputs as behavioral, and those outside the bin as non-behavioral. This aids in
    answering the question "Which $X_{i}$ contributes most toward a given range of outputs?".
    Factor space can also be assessed (`target="X"`), such that $f(X_{i}|b_{i}) -> (Y|b_{i})$
    and $f(X_{i}|b_{~i}) -> (Y|b_{~i})$. This aids in answering the question "where in factor
    space are outputs most sensitive to?"

    The $k$-samples Anderson-Darling test is used to compare distributions.
    Results of the analysis are normalized so that values will be $\\in [0, 1]$, and indicate
    relative sensitivity across factor space. Larger values indicate greater dissimilarity
    (thus, sensitivity).

    Notes
    -----
    Compatible with:
        all samplers

    When applied to grouped factors, the analysis is conducted on each factor individually,
    and the mean of the results for a group are reported.

    Increasing the value of \$S\$ increases the granularity of the analysis (across factor space),
    but necessitates larger sample sizes.

    This analysis will produce NaNs, indicating areas of factor space that did not have
    any samples, or for which the outputs were constant.

    Parameters
    ----------
    problem : dict
        The problem definition
    X : numpy.array
        A NumPy array containing the model inputs
    Y : numpy.array
        A NumPy array containing the model outputs
    bins : int
        The number of bins to use (default: 20)
    target : str
        Assess factor space ("X") or output space ("Y")
        (default: "Y")
    print_to_console : bool
        Print results directly to console (default False)
    seed : int
        Seed value to ensure deterministic results
        Unused, but defined to maintain compatibility.

    References
    ----------
    1. Pianosi, F., K. Beven, J. Freer, J. W. Hall, J. Rougier, D. B. Stephenson, and
    T. Wagener. 2016.
    Sensitivity analysis of environmental models:
    A systematic review with practical workflow.
    Environmental Modelling & Software 79:214-232.
    https://dx.doi.org/10.1016/j.envsoft.2016.02.008

    2. Saltelli, A., M. Ratto, T. Andres, F. Campolongo, J. Cariboni, D. Gatelli,
    M. Saisana, and S. Tarantola. 2008.
    Global Sensitivity Analysis: The Primer.
    Wiley, West Sussex, U.K.
    https://dx.doi.org/10.1002/9780470725184
    Accessible at: http://www.andreasaltelli.eu/file/repository/Primer_Corrected_2022.pdf
    """
    groups = _check_groups(problem)
    if not groups:
        var_names = problem["names"]
    else:
        var_names, _ = extract_group_names(problem.get("groups", []))

    results = rsa(X, Y, bins, target)

    if groups:
        groups = np.array(groups)
        unique_grps = [*dict.fromkeys(groups)]
        tmp = np.full((bins, len(unique_grps)), np.nan)

        # Take the mean of effects from parameters that are grouped together
        for grp_id, grp in enumerate(unique_grps):
            tmp[:, grp_id] = np.nanmean(results[:, groups == grp], axis=1)

        results = tmp
        tmp = None

    Si = ResultDict([(g, results[:, i]) for i, g in enumerate(var_names)])
    Si["names"] = var_names
    Si["bins"] = np.arange(0.0, 1.0, (1.0 / bins))
    Si["target"] = target

    # Attach object methods specific to this sensitivity method to ResultDict.
    Si.to_df = MethodType(to_df, Si)
    Si._plot = Si.plot
    Si.plot = MethodType(plot, Si)

    if print_to_console:
        print(Si.to_df())

    return Si


def rsa(X: np.ndarray, y: np.ndarray, bins: int = 10, target="X") -> np.ndarray:
    N, D = X.shape

    # Pre-allocated arrays to store data/results
    seq = np.append(np.arange(0.0, 1.0, (1 / bins)), 1.0)
    X_di = np.zeros(N)
    X_q = np.zeros(bins + 1)
    r_s = np.full((bins, D), np.nan)
    sel = np.zeros(N, dtype=bool)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for d_i in range(D):
            X_di[:] = X[:, d_i]

            if target == "X":
                X_q[:] = np.quantile(X_di, seq)
                sel[:] = (X_q[0] <= X_di) & (X_di <= X_q[1])
            elif target == "Y":
                y_q = np.quantile(y, seq)
                sel[:] = (y_q[0] <= y) & (y <= y_q[1])

            if (
                (np.count_nonzero(sel) != 0)
                and (len(y[~sel]) != 0)
                and np.unique(y[sel]).size > 1
            ):
                if target == "X":
                    r_s[0, d_i] = anderson_ksamp((y[sel], y[~sel])).statistic
                elif target == "Y":
                    r_s[0, d_i] = anderson_ksamp((X_di[sel], X_di[~sel])).statistic

            for s in range(1, bins):
                if target == "X":
                    sel[:] = (X_q[s] < X_di) & (X_di <= X_q[s + 1])
                elif target == "Y":
                    sel[:] = (y_q[s] <= y) & (y <= y_q[s + 1])

                if (
                    (np.count_nonzero(sel) == 0)
                    or (len(y[~sel]) == 0)
                    or np.unique(y[sel]).size == 1
                ):
                    continue

                if target == "X":
                    r_s[s, d_i] = anderson_ksamp((y[sel], y[~sel])).statistic
                elif target == "Y":
                    r_s[s, d_i] = anderson_ksamp((X_di[sel], X_di[~sel])).statistic

    min_val = np.nanmin(r_s)
    return (r_s - min_val) / (np.nanmax(r_s) - min_val)


def to_df(self):
    """Conversion to Pandas DataFrame specific to Regional Sensitivity Analysis results.

    Overrides the conversion method attached to the SALib problem spec.

    Returns
    -------
    Pandas DataFrame
    """
    names = self["names"]

    # Only convert these elements in dict to DF
    new_spec = {k: v for k, v in self.items() if k not in ["names", "bins", "target"]}

    return pd.DataFrame(new_spec, columns=names, index=self["bins"])


def plot(self, factors=None):
    """Plotting for Regional Sensitivity Analysis.

    Overrides the plot method attached to the SALib problem spec.
    """
    import matplotlib.pyplot as plt

    Si_df = self.to_df()
    fig, ax = plt.subplots()

    xlabel = f"${self['target']}$ (Percentile)"
    ylabel = "Relative $S_{i}$"

    if factors is None:
        factors = slice(None)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ax = Si_df.loc[:, factors].plot(
            ax=ax,
            xlabel=xlabel,
            ylabel=ylabel,
            subplots=True,
            legend=False,
            sharey=True,
            sharex=True,
        )

    fig.legend(bbox_to_anchor=(1.1, 1.0))
    fig.tight_layout()

    return ax


def cli_parse(parser):
    parser.add_argument(
        "-X", "--model-input-file", type=str, required=True, help="Model input file"
    )

    parser.add_argument(
        "-b",
        "--bins",
        type=int,
        required=False,
        help="Number of bins to partition target space into",
    )

    parser.add_argument(
        "-t",
        "--target",
        type=str,
        required=False,
        help="Target input space (X) or output space (Y)",
    )

    return parser


def cli_action(args):
    problem = read_param_file(args.paramfile)
    X = np.loadtxt(args.model_input_file, delimiter=args.delimiter)
    Y = np.loadtxt(
        args.model_output_file, delimiter=args.delimiter, usecols=(args.column,)
    )
    analyze(
        problem,
        X,
        Y,
        bins=args.bins,
        target=args.target,
        print_to_console=True,
        seed=args.seed,
    )


if __name__ == "__main__":
    common_args.run_cli(cli_parse, cli_action)
