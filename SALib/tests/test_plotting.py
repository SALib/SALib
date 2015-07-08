'''
Created on 29 Jun 2015

@author: will2

A useful guide as to how to get image_comparison working on your local machine
http://www.davidketcheson.info/2015/01/13/using_matplotlib_image_comparison.html
'''
import matplotlib
matplotlib.use('agg')
import numpy as np
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt

from SALib.plotting.morris import horizontal_bar_plot, \
                                  covariance_plot, \
                                  sample_histograms

@image_comparison(baseline_images=['horizontal_bar_plot'],
                  extensions=['png'])
def test_morris_horizontal_bar_plot():
    # Create a fixture to represent Si, the dictionary of output metrics
    
    Si = {'mu_star':[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          'names':['x1', 'x2', 'x3', 'x4', 'x5', 'x6'],
          'mu_star_conf':[0.5, 1, 1.5, 2, 2.5, 3.0],
          'sigma':[0.5, 1, 1.5, 2, 2.5, 3.0]} 
    
    fig, ax = plt.subplots(1, 1)
    horizontal_bar_plot(ax, Si,{}, sortby='mu_star', unit=r"tCO$_2$/year")


@image_comparison(baseline_images=['covariance_plot'],
                  extensions=['png'])
def test_morris_covariance_plot():
    # Create a fixture to represent Si, the dictionary of output metrics
    Si = {'mu_star':[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          'names':['x1', 'x2', 'x3', 'x4', 'x5', 'x6'],
          'mu_star_conf':[0.5, 1, 1.5, 2, 2.5, 3.0],
          'sigma':[0.5, 1, 1.5, 2, 2.5, 3.0]} 
    fig, ax = plt.subplots(1, 1)
    covariance_plot(ax, Si, {}, unit=r"tCO$_2$/year")

@image_comparison(baseline_images=['sample_histograms'],
                  extensions=['png'])
def test_morris_sample_histograms():
    
    input_1 = [[0, 1 / 3.], [0, 1.], [2 / 3., 1.]]
    input_2 = [[0, 1 / 3.], [2 / 3., 1 / 3.], [2 / 3., 1.]]
    input_3 = [[2 / 3., 0], [2 / 3., 2 / 3.], [0, 2 / 3.]]
    input_4 = [[1 / 3., 1.], [1., 1.], [1, 1 / 3.]]
    input_5 = [[1 / 3., 1.], [1 / 3., 1 / 3.], [1, 1 / 3.]]
    input_6 = [[1 / 3., 2 / 3.], [1 / 3., 0], [1., 0]]
    input_sample = np.concatenate([input_1, input_2, input_3,
                                   input_4, input_5, input_6])
    problem = {'num_vars': 2, 'groups': None, 'names':['x1','x2']}
    
    fig2 = plt.figure()
    sample_histograms(fig2, input_sample, problem, {'color':'y'})