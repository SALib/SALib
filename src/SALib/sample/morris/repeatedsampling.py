# -*- coding: utf-8 -*-
"""
Created on Thu May 23 17:08:07 2019

@author: engelen
"""

from itertools import combinations
import numpy as np
from . strategy import Strategy

class RepeatedSampling(Strategy):
    """Hacked together this strategy, I had to work around some of the checks.
    """
    @classmethod
    def run_checks(self, number_samples, k_choices):
        #HACK TO SKIP RUN CHECKS
        pass

    @classmethod
    def check_input_sample(self,input_sample, num_params, num_samples):
        pass

    def _sample(self, input_sample, num_samples,
                num_params, k_choices, num_groups):
        #Notice that we replace num_samples with sample4uniformity and k_choices with num_samples
        return self.find_most_distant(input_sample, num_samples, 
                          num_params, k_choices, num_groups)
    
    def find_most_distant(self, input_sample, sample4uniformity,
                          num_params, num_samples, num_groups):
        
        D = num_params
        
        distances = np.zeros(sample4uniformity)
        
        ranges = np.arange(num_samples * (D + 1))
        
        for i in range(sample4uniformity):
            index_list = ranges + i * num_samples * (D + 1)
            distance_matrix = self.compute_distance_matrix(input_sample[index_list, :], 
                                                           num_samples, num_params, local_optimization=True)
            distances[i] = self.sum_distances(np.arange(num_samples), distance_matrix)
        
        max_ind = np.argsort(distances)[-1:]
        index_list = ranges + max_ind * num_samples * (D+1)

        return list(index_list)

    def compile_output(self, input_sample, num_samples, num_params,
                       maximum_combo, num_groups=None):
        """Picks the trajectories from the input

        Arguments
        ---------
        input_sample : numpy.ndarray
        num_samples : int
        num_params : int
        maximum_combo : list
        num_groups : int

        """

        if num_groups is None:
            num_groups = num_params
            
        return(input_sample[maximum_combo, :])

    def sum_distances(self, indices, distance_matrix):
        """Calculate combinatorial distance between a select group of
        trajectories, indicated by indices

        Arguments
        ---------
        indices : tuple
        distance_matrix : numpy.ndarray (M,M)

        Returns
        -------
        numpy.ndarray

        Notes
        -----
        This function can perhaps be quickened by calculating the sum of the
        distances. The calculated distances, as they are right now,
        are only used in a relative way. Purely summing distances would lead
        to the same result, at a perhaps quicker rate.
        """
        combs_tup = combinations(indices, 2)

        combs = np.array(tuple(zip(*combs_tup)))

        # Calculate distance (vectorized)
        dist = np.sqrt(
            np.sum(np.square(distance_matrix[combs[0], combs[1]]), axis=0))

        return dist