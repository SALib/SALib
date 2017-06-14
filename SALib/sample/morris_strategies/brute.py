"""
"""
from SALib.sample.morris_strategies import Strategy
from .. morris_util import (find_maximum, compute_distance_matrix)
from scipy.misc import comb as nchoosek
from itertools import combinations, islice
import sys
import numpy as np


class BruteForce(Strategy):
    """Implements the brute force optimisation strategy
    """

    def _sample(self, input_sample, num_samples,
                num_params, k_choices, groups):
        return self.brute_force_most_distant(input_sample, num_samples,
                                             num_params, k_choices, groups)

    def brute_force_most_distant(self, input_sample, num_samples,
                                 num_params, k_choices,
                                 groups=None):
        """Use brute force method to find most distant trajectories

        Arguments
        ---------
        input_sample : numpy.ndarray
        num_samples : int
            The number of samples to generate
        num_params : int
            The number of parameters
        k_choices : int
            The number of optimal trajectories
        groups : tuple
            A tuple of (numpy.ndarray, list)
        """
        scores = self.find_most_distant(input_sample,
                                        num_samples,
                                        num_params,
                                        k_choices,
                                        groups)

        maximum_combo = find_maximum(scores, num_samples, k_choices)

        return maximum_combo

    def find_most_distant(self, input_sample, num_samples,
                          num_params, k_choices, groups=None):
        """
        Finds the 'k_choices' most distant choices from the
        'num_samples' trajectories contained in 'input_sample'

        Arguments
        ---------
        input_sample : numpy.ndarray
        num_samples : int
            The number of samples to generate
        num_params : int
            The number of parameters
        k_choices : int
            The number of optimal trajectories
        groups : tuple
            A tuple of (numpy.ndarray, list)

        Returns
        -------
        numpy.ndarray
        """
        # Now evaluate the (N choose k_choices) possible combinations
        if nchoosek(num_samples, k_choices) >= sys.maxsize:
            raise ValueError("Number of combinations is too large")
        number_of_combinations = int(nchoosek(num_samples, k_choices))

        # First compute the distance matrix for each possible pairing
        # of trajectories and store in a shared-memory array
        distance_matrix = compute_distance_matrix(input_sample,
                                                  num_samples,
                                                  num_params,
                                                  groups)

        # Initialise the output array
        chunk = int(1e6)
        if chunk > number_of_combinations:
            chunk = number_of_combinations

        counter = 0
        # Generate a list of all the possible combinations
        combo_gen = combinations(list(range(num_samples)), k_choices)
        scores = np.zeros(number_of_combinations, dtype=np.float32)
        # Generate the pairwise indices once
        pairwise = np.array(
            [y for y in combinations(list(range(k_choices)), 2)])

        for combos in self.grouper(chunk, combo_gen):
            scores[(counter * chunk):((counter + 1) * chunk)] \
                = self.mappable(combos, pairwise, distance_matrix)
            counter += 1
        return scores

    @staticmethod
    def grouper(n, iterable):
        it = iter(iterable)
        while True:
            chunk = tuple(islice(it, n))
            if not chunk:
                return
            yield chunk

    @staticmethod
    def mappable(combos, pairwise, distance_matrix):
        '''
        Obtains scores from the distance_matrix for each pairwise combination
        held in the combos array
        '''
        combos = np.array(combos)
        # Create a list of all pairwise combination for each combo in combos
        combo_list = combos[:, pairwise[:, ]]
        all_distances = distance_matrix[[
            combo_list[:, :, 1], combo_list[:, :, 0]]]
        new_scores = np.sqrt(
            np.einsum('ij,ij->i', all_distances, all_distances))
        return new_scores
