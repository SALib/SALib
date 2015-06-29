'''
Created on 29 Jun 2015

@author: will2

A useful guide as to how to get image_comparison working on your local machine
http://www.davidketcheson.info/2015/01/13/using_matplotlib_image_comparison.html
'''
import numpy as np
import matplotlib
from matplotlib.testing.decorators import image_comparison
import matplotlib.pyplot as plt

from SALib.plotting import morris

@image_comparison(baseline_images=['morris_horiz_bar_plot'])
def test_morris_horizontal_bar_plot():
    # Create a fixture to represent Si, the dictionary of output metrics
    
    Si = {'mu_star':[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
          'names':['x1', 'x2', 'x3', 'x4', 'x5', 'x6'],
          'mu_star_conf':[0.5, 1, 1.5, 2, 2.5, 3.0]} 
    
    morris.horiz_bar_plot(Si, sortby='mu_star')
#     plt.savefig('morris_horiz_bar_plot.png')