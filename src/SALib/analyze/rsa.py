from typing import Dict
from types import MethodType
import warnings

import numpy as np
import pandas as pd
from scipy.stats import cramervonmises_2samp

from . import common_args
from ..util import read_param_file, ResultDict, extract_group_names


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

    In a usual RSA, a desirable region of output space is defined. Outputs which fall
    within this region is categorized as being "behavioral" (:math:`B`), and those
    outside are described as being "non-behavioral" (:math:`\\bar{B}`). The input
    factors are also partitioned into behavioral and non-behavioral subsets, such that
    :math:`f(X_{i}|B) \\rightarrow (Y|B)` and :math:`f(X_{i}|\\bar{B}) \\rightarrow
    (Y|\\bar{B})`. The distribution between the two sub-samples are compared for each
    factor. The greater the difference between the two distributions, the more
    important the given factor is in driving model outputs.

    The approach implemented in SALib partitions factor or output space into :math:`b`
    bins (default: 20) according to their percentile values. Output space is targeted
    for analysis by default (:code:`target="Y"`), such that :math:`(Y|b_{i})` is
    mapped back to :math:`(X_{i}|b_{i})`. In other words, we treat outputs falling
    within a given bin (:math:`b_{i}`) corresponding to their inputs as behavioral, and
    those outside the bin as non-behavioral. This aids in answering the question
    "Which :math:`X_{i}` contributes most toward a given range of outputs?". Factor
    space can also be assessed (:code:`target="X"`), such that :math:`f(X_{i}|b_{i})
    \\rightarrow (Y|b_{i})` and :math:`f(X_{i}|b_{\\sim i}) \\rightarrow
    (Y|b_{\\sim i})`. This aids in answering the question "where in factor space are
    outputs most sensitive to?"

    The two-sample Cramér-von Mises (CvM) test is used to compare distributions.
    Results of the analysis indicate sensitivity across factor/output space. As the
    Cramér-von Mises criterion ranges from 0 to :math:`\\infty`, a value of zero will
    indicates the two distributions being compared are identical, with larger values
    indicating greater differences.

    Notes
    -----
    Compatible with:
        all samplers

    When applied to grouped factors, the analysis is conducted on each factor
    individually, and the mean of the results for a group are reported.

    Increasing the value of :code:`bins` increases the granularity of the analysis
    (across factor space), but necessitates larger sample sizes.

    This analysis will produce NaNs, indicating areas of factor space that did not have
    any samples, or for which the outputs were constant.

    Analysis results are normalized against the maximum value such that 1.0 indicates
    the greatest sensitivity.

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
    1. Hornberger, G. M., and R. C. Spear. 1981.
        Approach to the preliminary analysis of environmental systems.
        Journal of Environmental Management 12:1.
        https://www.osti.gov/biblio/6396608-approach-preliminary-analysis-environmental-systems

    2. Pianosi, F., K. Beven, J. Freer, J. W. Hall, J. Rougier, D. B. Stephenson, and
        T. Wagener. 2016.
        Sensitivity analysis of environmental models:
        A systematic review with practical workflow.
        Environmental Modelling & Software 79:214-232.
        https://dx.doi.org/10.1016/j.envsoft.2016.02.008

    3. Saltelli, A., M. Ratto, T. Andres, F. Campolongo, J. Cariboni, D. Gatelli,
        M. Saisana, and S. Tarantola. 2008.
        Global Sensitivity Analysis: The Primer.
        Wiley, West Sussex, U.K.
        https://dx.doi.org/10.1002/9780470725184
        Accessible at:
        http://www.andreasaltelli.eu/file/repository/Primer_Corrected_2022.pdf
    """
    var_names, _ = extract_group_names(problem)

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
    X_di = np.empty(N)  # store of factor values
    r_s = np.full((bins, D), np.nan)  # results

    if target == "X":
        t_arr = X_di  # target factor space for analysis
        m_arr = y  # map behavioral region of factor space to output space
    elif target == "Y":
        t_arr = y  # target output space for analysis
        m_arr = X_di  # map outputs back to factor space

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for d_i in range(D):
            X_di[:] = X[:, d_i]

            # Assess first bin separately, making sure that the
            # bin edges are inclusive of the lower bound
            quants = np.quantile(t_arr, seq)
            b = (quants[0] <= t_arr) & (t_arr <= quants[1])
            if _has_samples(y, b):
                r_s[0, d_i] = cramervonmises_2samp(m_arr[b], m_arr[~b]).statistic

            # Then assess the other bins
            for s in range(1, bins):
                b = (quants[s] < t_arr) & (t_arr <= quants[s + 1])

                if _has_samples(y, b):
                    r_s[s, d_i] = cramervonmises_2samp(m_arr[b], m_arr[~b]).statistic

    return r_s


def _has_samples(y, sel):
    """Check if the given region of factor space has > 0 samples.

    Returns
    -------
    bool, true if > 0 non-unique samples are found, false otherwise.
    """
    return (
        (np.count_nonzero(sel) != 0)
        and (len(y[~sel]) != 0)
        and np.unique(y[sel]).size > 1
    )


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
    ylabel = "$CvM_{i}$"

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
