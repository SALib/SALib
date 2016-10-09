'''
Created on 29 Jun 2015

@author: @willu47

This module provides the basic infrastructure for plotting charts for the
Method of Morris results

The procedures should build upon and return an axes instance.

Si = morris.analyze(problem, param_values, Y, conf_level=0.95, print_to_console=False, num_levels=10, grid_jump=5)
p = morris.horizontal_bar_plot(Si)
# set plot style etc.

fig, ax = plt.subplots(1, 1)
my_plotter(ax, data1, data2, {'marker':'x'})

p.show()

def my_plotter(ax, data1, data2, param_dict):
    A helper function to make a graph

    Parameters
    ax : Axes
        The axes to draw to

    data1 : array
       The x data

    data2 : array
       The y data

    param_dict : dict
       Dictionary of kwargs to pass to ax.plot

    Returns
    out : list
        list of artists added
    """
    out = ax.plot(data1, data2, **param_dict)
    return out

'''

import matplotlib.pyplot as plt
import numpy as np

def _sort_Si(Si, key, sortby='mu_star'):
    return np.array([Si[key][x] for x in np.argsort(Si[sortby])])


def _sort_Si_by_index(Si, key, index):
    return np.array([Si[key][x] for x in index])


def horizontal_bar_plot(ax, Si, param_dict, sortby='mu_star', unit=''):
    '''Updates a matplotlib axes instance with a horizontal bar plot

    of mu_star, with error bars representing mu_star_conf
    '''

    assert sortby in ['mu_star', 'mu_star_conf', 'sigma', 'mu']

    fig = ax.get_figure()

    # Sort all the plotted elements by mu_star (or optionally another
    # metric)
    names_sorted = _sort_Si(Si, 'names', sortby)
    mu_star_sorted = _sort_Si(Si, 'mu_star', sortby)
    mu_star_conf_sorted = _sort_Si(Si, 'mu_star_conf', sortby)

    # Plot horizontal barchart
    y_pos = np.arange(len(mu_star_sorted))
    plot_names = names_sorted

    out = ax.barh(y_pos,
                  mu_star_sorted,
                  xerr=mu_star_conf_sorted,
                  align='center',
                  ecolor='black',
                  **param_dict)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_names)
    ax.set_xlabel(r'$\mu^\star$' + unit)

    ax.set_ylim(min(y_pos)-1, max(y_pos)+1)

    return out


def covariance_plot(ax, Si, param_dict, unit=""):
    '''Plots mu* against sigma or the 95% confidence interval

    '''

    if Si['sigma'] is not None:
        # sigma is not present if using morris groups
        y = Si['sigma']
        out = ax.scatter(Si['mu_star'], y, c=u'k', marker=u'o',
                 **param_dict)
        ax.set_ylabel(r'$\sigma$')

        ax.set_xlim(0,)
        ax.set_ylim(0,)

        x_axis_bounds = np.array(ax.get_xlim())

        line1, = ax.plot(x_axis_bounds, x_axis_bounds, 'k-')
        line2, = ax.plot(x_axis_bounds, 0.5 * x_axis_bounds, 'k--')
        line3, = ax.plot(x_axis_bounds, 0.1 * x_axis_bounds, 'k-.')

        ax.legend((line1, line2, line3), (r'$\sigma / \mu^{\star} = 1.0$',
                                          r'$\sigma / \mu^{\star} = 0.5$',
                                          r'$\sigma / \mu^{\star} = 0.1$'),
                  loc='best')

    else:
        y = Si['mu_star_conf']
        out = ax.scatter(Si['mu_star'], y, c=u'k', marker=u'o',
                 **param_dict)
        ax.set_ylabel(r'$95\% CI$')

    ax.set_xlabel(r'$\mu^\star$ ' + unit)
    ax.set_ylim(0-(0.01 * np.array(ax.get_ylim()[1])), )

    return out


def sample_histograms(fig, input_sample, problem, param_dict):
    '''Plots a set of subplots of histograms of the input sample

    '''
    num_vars = problem['num_vars']
    names = problem['names']

    framing = 101 + (num_vars * 10)

    # Find number of levels
    num_levels = len(set(input_sample[:,1]))

    out = []

    for variable in range(num_vars):
        ax = fig.add_subplot(framing + variable)
        out.append(ax.hist(input_sample[:, variable],
                           bins=num_levels,
                           normed=False,
                           label=None,
                           **param_dict))

        ax.set_title('%s' % (names[variable]))
        ax.tick_params(axis='x',  # changes apply to the x-axis
                       which='both',  # both major and minor ticks are affected
                       bottom='off',  # ticks along the bottom edge are off
                       top='off',  # ticks along the top edge are off
                       labelbottom='off')  # labels along the bottom edge are off)
        if variable > 0:
            ax.tick_params(axis='y',  # changes apply to the y-axis
                           which='both',  # both major and minor ticks are affected
                           labelleft='off')  # labels along the left edge are off)

    return out

if __name__ == '__main__':
    pass
