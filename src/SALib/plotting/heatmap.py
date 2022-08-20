from typing import Iterable
import numpy as np
import matplotlib.pyplot as plt


__all__ = ['heatmap']


# magic string indicating DF columns holding conf bound values
CONF_COLUMN = '_conf'


def heatmap_Y(sp, index: str, metrics: list = None) -> np.ndarray:
    """Produce data for heatmap plot for a single sensitivity index across one
    or more outputs."""
    if isinstance(metrics, str):
        metrics = [metrics]

    if len(sp['outputs']) == 1:
        fig_data = np.array([sp.analysis[index]])
    else:
        fig_data = np.array([sp.analysis[index][m] for m in metrics])

    return fig_data


def heatmap_Si(sp, metric: str, indices: list = None) -> np.ndarray:
    """Produce data for heatmap plot for a single output across one
    or more sensitivity indices."""
    if isinstance(indices, str):
        indices = [indices]

    if len(sp['outputs']) > 1:
        fig_data = np.array([sp.analysis[idx][metric] for idx in indices])
    else:
        si_shape = sp.analysis[indices[0]].shape
        fig_data = np.array([sp.analysis[idx] for idx in indices
                             if sp.analysis[idx].shape == si_shape])
    return fig_data


def heatmap(sp, metric, index, title: str = None, ax=None):
    """Plot a heatmap of the target metric.

    Parameters
    ----------
    sp : object, SALib ProblemSpec
    metric : str, metric to plot. Defaults to first metric/result output if not given.
    index : str, sensitivity indices to plot ('S1', 'ST', etc)
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
        metric = sp['outputs'][0]

    if isinstance(metric, str):
        assert metric in sp['outputs'], f"Specified model output '{metric}' not found"

    if isinstance(metric, str) and isinstance(index, str):
        if len(sp['outputs']) == 1:
            res_display = np.array([sp.analysis[index]])
        else:
            res_display = np.array([sp.analysis[metric][index]])
    else:
        if isinstance(index, (list, tuple)) or index is None:
            if index is None:
                index = [k for k in sp.analysis.keys()
                         if not k.endswith(CONF_COLUMN) and k != "names"]

            res_display = heatmap_Si(sp, metric, index)
        elif isinstance(metric, (list, tuple)) or metric is None:
            if metric is None:
                metric = sp["outputs"]
            res_display = heatmap_Y(sp, index, metric)

    is_idx_def = isinstance(index, (str, list, tuple))

    ax.imshow(res_display)
    fig.colorbar(ax.images[0], ax=ax, shrink=0.9)

    if title is None:
        if is_idx_def:
            d_met = metric
            if isinstance(d_met, (list, tuple)):
                d_met = d_met[0]
            # Use generic title if metric not defined
            title = f"Sensitivity of {d_met}"
        else:
            title = metric

    ax.set_title(title)

    if isinstance(metric, str):
        metric = [metric]
    if isinstance(index, str):
        index = [index]

    # Get unique groups (if defined) while maintaining order of group names
    disp_names = sp['groups']
    if disp_names is None:
        disp_names = sp['names']

    ax.xaxis.set_ticks(range(0, len(disp_names)))
    ax.xaxis.set_ticklabels(disp_names, rotation=90)

    # Account for indices that may have been filtered out
    # e.g., cannot easily show second-order values at the moment
    disp_idx = res_display.shape[0]
    if is_idx_def:
        ax.yaxis.set_ticks(range(0, len(index[0:disp_idx])))
        ax.yaxis.set_ticklabels(index[0:disp_idx])
    else:
        ax.yaxis.set_ticks(range(0, len(metric)))
        ax.yaxis.set_ticklabels(metric)

    return ax
