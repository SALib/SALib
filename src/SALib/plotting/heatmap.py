import numpy as np
import matplotlib.pyplot as plt


__all__ = ['heatmap']


# magic string indicating DF columns holding conf bound values
CONF_COLUMN = '_conf'


def heatmap(sp, metric: str, title: str = None, ax=None):
    """Plot a heatmap of the target metric.

    Parameters
    ----------
    sp : object, SALib ProblemSpec
    metric : str, metric to target ('S1', 'ST', etc)
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

    if len(sp['outputs']) > 1:
        res_display = np.array([sp.analysis[out][metric]
                                for out in sp['outputs']])
    else:
        res_display = np.array([sp.analysis[metric]])

    ax.imshow(res_display)
    fig.colorbar(ax.images[0], ax=ax, shrink=0.9)

    if title is None:
        title = metric

    ax.set_title(title)
    
    ax.xaxis.set_ticks(range(0, len(sp['names'])))
    ax.xaxis.set_ticklabels(sp['names'], rotation=90)

    ax.yaxis.set_ticks(range(0, len(sp['outputs'])))
    ax.yaxis.set_ticklabels(sp['outputs'])

    return ax
