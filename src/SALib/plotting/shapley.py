"""Bar chart for Shapley effects.

Raw Shapley effects (output-variance units) and their normalized shares
(unit interval, summing to one) sit on incompatible scales, so plotting all
four ``shapley``/``shapley_conf``/``shapley_normalized``/
``shapley_normalized_conf`` columns together in a single bar chart (the
default behavior inherited from ``ResultDict.plot``) makes the smaller
series unreadable. This module plots one pair at a time instead, defaulting
to the normalized shares since those are what's usually of interest.
"""

import matplotlib.pyplot as plt

from .bar import plot as barplot

__all__ = ["plot"]


def plot(Si, ax=None, normalized=True):
    """Plot Shapley effects as a bar chart.

    Parameters
    ----------
    Si : ResultDict
        Analysis results, as returned by :func:`SALib.analyze.shapley.analyze`.
    ax : matplotlib axes object, optional
        Axes to plot onto. Creates a new figure if not provided.
    normalized : bool, default=True
        Plot the normalized shares (summing to one) rather than the raw
        effects in output-variance units.

    Returns
    -------
    ax : matplotlib axes object
    """
    df = Si.to_df()

    if normalized:
        cols = ["shapley_normalized", "shapley_normalized_conf"]
        title = "Normalized Shapley effects"
    else:
        cols = ["shapley", "shapley_conf"]
        title = "Shapley effects"

    if ax is None:
        _, ax = plt.subplots()

    barplot(df[cols], ax=ax)
    ax.set_title(title)

    return ax
