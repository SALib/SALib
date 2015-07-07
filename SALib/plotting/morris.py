'''
Created on 29 Jun 2015

@author: @willu47

This module provides the basic infrastructure for plotting charts for the 
Method of Morris results

Si = morris.analyze(problem, param_values, Y, conf_level=0.95, print_to_console=False, num_levels=10, grid_jump=5)
p = morris.horizontal_bar_plot(Si)
# set plot style etc.
p.show()

'''

import matplotlib.pyplot as plt
import numpy as np

def _sort_Si(Si, key, sortby='mu_star'):
    return np.array([Si[key][x] for x in np.argsort(Si[sortby])])


def _sort_Si_by_index(Si, key, index):
    return np.array([Si[key][x] for x in index])


def horizontal_bar_plot(Si, sortby='mu_star'):
    '''
    Returns a matplotlib plot instance
    Use p.show() to display the plot
    '''
    
    assert sortby in ['mu_star', 'mu_star_conf', 'sigma', 'mu']
    
    # Sort all the plotted elements by mu_star (or optionally another
    # metric)
    names_sorted = _sort_Si(Si, 'names', sortby)
    mu_star_sorted = _sort_Si(Si, 'mu_star', sortby)
    mu_star_conf_sorted = _sort_Si(Si, 'mu_star_conf', sortby)
    
    # Plot horizontal barchart
    y_pos = np.arange(len(mu_star_sorted))
    plot_names = names_sorted
    plot_mu_star = mu_star_sorted
    plot_mu_star_conf = mu_star_conf_sorted

    plt.barh(y_pos, plot_mu_star, xerr=plot_mu_star_conf, align='center', ecolor='black')

    plt.yticks(y_pos , plot_names)
    plt.xlabel('mu*')
    plt.title('Morris Screening Analysis')
    plt.tight_layout()

    return plt


def covariance_plot(Si, unit=""):
    '''
    Plots mu* against sigma or the 95% confidence interval
    '''
    plt.scatter(Si['mu_star'], Si['mu_star_conf'])
    plt.xlabel('mu* ' + unit)
    plt.ylabel('95% CI')
#     plt.title(results_prefix)
    plt.tight_layout()
    
    return plt


def sample_histograms(input_sample, problem):
    '''
    Plots a set of subplots of histograms of the input sample
    '''
    num_vars = problem['num_vars']
    names = problem['names']
    
    
    fig, axes = plt.subplots(nrows=1, ncols=num_vars)
    
    for v in range(num_vars):
        axes[v].hist(input_sample[:, v], bins=4, normed=False)
        axes[v].set_title('%s' % (names[v]))
        axes[v].xaxis.set_tick_params(label1On=False)
        axes[v].yaxis.set_tick_params(label1On=True)    
    
    plt.tight_layout()
    
    return plt





if __name__ == '__main__':
    pass
