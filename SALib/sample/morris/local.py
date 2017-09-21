"""
"""
from itertools import combinations
import numpy as np

from . strategy import Strategy


class LocalOptimisation(Strategy):
    """Implements the local optimisation algorithm using the Strategy interface
    """

    def _sample(self, input_sample, num_samples,
                num_params, k_choices, num_groups=None):
        return self.find_local_maximum(input_sample, num_samples, num_params,
                                       k_choices, num_groups)

    def find_local_maximum(self, input_sample, N, num_params,
                           k_choices, num_groups=None):
        """Find the most different trajectories in the input sample using a
        local approach

        An alternative by Ruano et al. (2012) for the brute force approach as
        originally proposed by Campolongo et al. (2007). The method should
        improve the speed with which an optimal set of trajectories is
        found tremendously for larger sample sizes.

        Arguments
        ---------
        input_sample : np.ndarray
        N : int
            The number of trajectories
        num_params : int
            The number of factors
        k_choices : int
            The number of optimal trajectories to return
        num_groups : int, default=None
            The number of groups
        Returns
        -------
        list
        """
        distance_matrix = self.compute_distance_matrix(input_sample, N,
                                                       num_params, num_groups,
                                                       local_optimization=True)

        tot_indices_list = []
        tot_max_array = np.zeros(k_choices - 1)

        # Loop over `k_choices`, i starts at 1
        for i in range(1, k_choices):
            indices_list = []
            row_maxima_i = np.zeros(len(distance_matrix))

            row_nr = 0
            for row in distance_matrix:
                indices = tuple(row.argsort()[-i:][::-1]) + (row_nr,)
                row_maxima_i[row_nr] = self.sum_distances(
                    indices, distance_matrix)
                indices_list.append(indices)
                row_nr += 1

            # Find the indices belonging to the maximum distance
            i_max_ind = self.get_max_sum_ind(indices_list, row_maxima_i, i, 0)

            # Loop 'm' (called loop 'k' in Ruano)
            m_max_ind = i_max_ind
            # m starts at 1
            m = 1

            while m <= k_choices - i - 1:
                m_ind = self.add_indices(m_max_ind, distance_matrix)

                m_maxima = np.zeros(len(m_ind))

                for n in range(0, len(m_ind)):
                    m_maxima[n] = self.sum_distances(m_ind[n], distance_matrix)

                m_max_ind = self.get_max_sum_ind(m_ind, m_maxima, i, m)

                m += 1

            tot_indices_list.append(m_max_ind)
            tot_max_array[i -
                          1] = self.sum_distances(m_max_ind, distance_matrix)

        tot_max = self.get_max_sum_ind(
            tot_indices_list, tot_max_array, "tot", "tot")
        return sorted(list(tot_max))

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
        combs_tup = np.array(tuple(combinations(indices, 2)))

        # Put indices from tuples into two-dimensional array.
        combs = np.array([[i[0] for i in combs_tup],
                          [i[1] for i in combs_tup]])

        # Calculate distance (vectorized)
        dist = np.sqrt(
            np.sum(np.square(distance_matrix[combs[0], combs[1]]), axis=0))

        return dist

    def get_max_sum_ind(self, indices_list, distances, i, m):
        '''Get the indices that belong to the maximum distance in `distances`

        Arguments
        ---------
        indices_list : list
            list of tuples
        distances : numpy.ndarray
            size M
        i : int
        m : int

        Returns
        -------
        list
        '''
        if len(indices_list) != len(distances):
            msg = "Indices and distances are lists of different length." + \
                "Length indices_list = {} and length distances = {}." + \
                "In loop i = {}  and m =  {}"
            raise ValueError(msg.format(
                len(indices_list), len(distances), i, m))

        max_index = tuple(distances.argsort()[-1:][::-1])
        return indices_list[max_index[0]]

    def add_indices(self, indices, distance_matrix):
        '''Adds extra indices for the combinatorial problem.

        Arguments
        ---------
        indices : tuple
        distance_matrix : numpy.ndarray (M,M)

        Example
        -------
        >>> add_indices((1,2), numpy.array((5,5)))
        [(1, 2, 3), (1, 2, 4), (1, 2, 5)]

        '''
        list_new_indices = []
        for i in range(0, len(distance_matrix)):
            if i not in indices:
                list_new_indices.append(indices + (i,))
        return list_new_indices
