from __future__ import division
import numpy as np
import random as rd
from . import common_args
from ..sample import morris_oat, morris_groups, morris_optimal
from ..util import read_param_file, scale_samples, read_group_file
from collections import Iterable

class Sample(object):
    '''
    A template class, which all of the sample classes inherit.
    '''

    def __init__(self, parameter_file, samples):

        self.parameter_file = parameter_file
        pf = read_param_file(self.parameter_file)
        self.num_vars = pf['num_vars']
        self.bounds = pf['bounds']
        self.parameter_names = pf['names']
        self.samples = samples
        self.output_sample = None


    def save_data(self, output, delimiter=' ', precision=8):
        '''
        Saves the data to a file for input into a model
        '''

        data_to_save = self.get_input_sample_scaled()

        np.savetxt(output,
                   data_to_save,
                   delimiter=delimiter,
                   fmt='%.' + str(precision) + 'e')


    def get_input_sample_unscaled(self):
        '''
        Returns the unscaled (according to the bounds from the parameter file)
        data as a numpy array
        '''
        return self.output_sample


    def get_input_sample_scaled(self):
        '''
        Returns the scaled (according to the bounds from the parameter file)
        data as a numpy array
        '''
        scaled_samples = self.output_sample.copy()
        scale_samples(scaled_samples, self.bounds)
        return scaled_samples


    def parameter_names(self):
        return self.names