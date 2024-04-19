import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


__all__ = ["heatmap"]


# magic string indicating DF columns holding conf bound values
CONF_COLUMN = "_conf"


def heatmap(sp, metric: str, index: str, title: str = None, ax=None):
    """Plot a heatmap of the target metric.

    Parameters
    ----------
    sp : object, SALib ProblemSpec
    metric : str, metric to plot. Defaults to first metric/result output if `None`.
    index : str, sensitivity indices to plot ('S1', 'ST', etc). Displays all if `None`.
    title : str, plot title to use
    ax : axes object, matplotlib axes object to assign figure to.

    Returns
    -------
    ax : matplotlib axes object
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    else:
        fig = plt.gcf()

    if metric is None:
        metric = sp["outputs"][0]

    if isinstance(metric, str):
        assert metric in sp["outputs"], f"Specified model output '{metric}' not found"

    is_multi_output = len(sp["outputs"]) > 1
    if not index:
        if is_multi_output:
            index = list(sp.analysis[metric].keys())
        else:
            index = list(sp.analysis.keys())

    multi_index = isinstance(index, (list, tuple))
    if multi_index:
        index = [k for k in index if not k.endswith(CONF_COLUMN) and k != "names"]

    if isinstance(index, str):
        index = [index]

    if is_multi_output:
        met_data = sp.analysis[metric]
    else:
        met_data = sp.analysis

    si_shape = met_data[index[0]].shape
    s_data = np.vstack([met_data[k] for k in index if met_data[k].shape == si_shape])

    ax.imshow(s_data)
    fig.colorbar(ax.images[0], ax=ax, shrink=0.9)

    if title is None:
        title = metric

    ax.set_title(title)

    if isinstance(metric, str):
        metric = [metric]

    # Get unique groups (if defined) while maintaining order of group names
    # Note: using pandas `unique` here as `numpy` sorts the values
    disp_names = pd.unique(sp["groups"])
    if disp_names is None:
        disp_names = sp["names"]

    ax.xaxis.set_ticks(range(0, len(disp_names)))
    ax.xaxis.set_ticklabels(disp_names, rotation=90)

    # Account for indices that may have been filtered out
    # e.g., cannot easily show second-order values at the moment
    disp_idx = s_data.shape[0]
    ax.yaxis.set_ticks(range(0, len(index[0:disp_idx])))
    ax.yaxis.set_ticklabels(index[0:disp_idx])

    return ax
