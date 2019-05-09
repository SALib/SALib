"""
Defines a family of algorithms for generating samples

The sample a for use with :class:`SALib.analyze.morris.analyze`,
encapsulate each one, and makes them interchangeable.

Example
-------
>>> localoptimisation = LocalOptimisation()
>>> context = SampleMorris(localoptimisation)
>>> context.sample(input_sample, num_samples, num_params, k_choices, groups)
"""
import abc

import numpy as np
from scipy.spatial.distance import cdist


class SampleMorris(object):
    """Computes the optimum `k_choices` of trajectories from the input_sample.

    Arguments
    ---------
    strategy : :class:`Strategy`
    """

    def __init__(self, strategy):
        self._strategy = strategy

    def sample(self, input_sample, num_samples, num_params, k_choices,
               num_groups):
        """Computes the optimum k_choices of trajectories
        from the input_sample.

        Arguments
        ---------
        input_sample : numpy.ndarray
        num_samples : int
            The number of samples to generate
        num_params : int
            The number of parameters
        k_choices : int
            The number of optimal trajectories
        num_groups : int
            The number of groups

        Returns
        -------
        numpy.ndarray
            An array of optimal trajectories
        """
        return self._strategy.sample(input_sample, num_samples, num_params,
                                     k_choices, num_groups)


class Strategy:
    """
    Declare an interface common to all supported algorithms.
    :class:`SampleMorris` uses this interface to call the algorithm
    defined by a ConcreteStrategy.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def _sample(self, input_sample, num_samples,
                num_params, k_choices, num_groups):
        """Implement this in your class

        Arguments
        ---------
        input_sample : numpy.ndarray
        num_samples : int
            The number of samples to generate
        num_params : int
            The number of parameters
        k_choices : int
            The number of optimal trajectories
        num_groups : int
            The number of groups

        Returns
        -------
        list
            A list of trajectory indices
        """
        pass

    def sample(self, input_sample, num_samples, num_params,
               k_choices, num_groups=None):
        """Computes the optimum k_choices of trajectories
        from the input_sample.

        Arguments
        ---------
        input_sample : numpy.ndarray
        num_samples : int
            The number of samples to generate
        num_params : int
            The number of parameters
        k_choices : int
            The number of optimal trajectories
        num_groups : int, default=None
            The number of groups

        Returns
        -------
        numpy.ndarray
        """
        self.run_checks(num_samples, k_choices)
        maximum_combo = self._sample(input_sample, num_samples,
                                     num_params, k_choices, num_groups)

        assert isinstance(maximum_combo, list)

        output = self.compile_output(input_sample,
                                     num_samples,
                                     num_params,
                                     maximum_combo,
                                     num_groups)
        return output

    @staticmethod
    def run_checks(number_samples, k_choices):
        """Runs checks on `k_choices`
        """
        assert isinstance(k_choices, int), \
            "Number of optimal trajectories should be an integer"

        if k_choices < 2:
            raise ValueError(
                "The number of optimal trajectories must be set to 2 or more.")
        if k_choices >= number_samples:
            msg = "The number of optimal trajectories should be less than the \
                    number of samples"
            raise ValueError(msg)

    @staticmethod
    def _make_index_list(num_samples, num_params, num_groups=None):
        """Identify indices of input sample associated with each trajectory

        For each trajectory, identifies the indexes of the input sample which
        is a function of the number of factors/groups and the number of samples

        Arguments
        ---------
        num_samples : int
            The number of trajectories
        num_params : int
            The number of parameters
        num_groups : int
            The number of groups

        Returns
        -------
        list of numpy.ndarray

        Example
        -------
        >>> BruteForce()._make_index_list(num_samples=4, num_params=3,
            num_groups=2)
        [np.array([0, 1, 2]), np.array([3, 4, 5]), np.array([6, 7, 8]),
         np.array([9, 10, 11])]
        """
        if num_groups is None:
            num_groups = num_params

        index_list = []
        for j in range(num_samples):
            index_list.append(np.arange(num_groups + 1) + j * (num_groups + 1))
        return index_list

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

        self.check_input_sample(input_sample, num_groups, num_samples)

        index_list = self._make_index_list(num_samples, num_params, num_groups)

        output = np.zeros(
            (np.size(maximum_combo) * (num_groups + 1), num_params))
        for counter, combo in enumerate(maximum_combo):
            output[index_list[counter]] = np.array(
                input_sample[index_list[combo]])
        return output

    @staticmethod
    def check_input_sample(input_sample, num_params, num_samples):
        """Check the `input_sample` is valid

        Checks input sample is:
            - the correct size
            - values between 0 and 1

        Arguments
        ---------
        input_sample : numpy.ndarray
        num_params : int
        num_samples : int
        """
        assert type(input_sample) == np.ndarray, \
            "Input sample is not an numpy array"
        assert input_sample.shape[0] == (num_params + 1) * num_samples, \
            "Input sample does not match number of parameters or groups"
        assert np.any((input_sample >= 0) | (input_sample <= 1)), \
            "Input sample must be scaled between 0 and 1"

    @staticmethod
    def compute_distance(m, l):
        '''Compute distance between two trajectories

        Returns
        -------
        numpy.ndarray
        '''

        if np.shape(m) != np.shape(l):
            raise ValueError("Input matrices are different sizes")
        if np.array_equal(m, l):
            # print("Trajectory %s and %s are equal" % (m, l))
            distance = 0
        else:
            distance = np.array(np.sum(cdist(m, l)), dtype=np.float32)

        return distance

    def compute_distance_matrix(self, input_sample, num_samples, num_params,
                                num_groups=None,
                                local_optimization=False):
        """Computes the distance between each and every trajectory

        Each entry in the matrix represents the sum of the geometric distances
        between all the pairs of points of the two trajectories

        If the `groups` argument is filled, then the distances are still
        calculated for each trajectory,

        Arguments
        ---------
        input_sample : numpy.ndarray
            The input sample of trajectories for which to compute
            the distance matrix
        num_samples : int
            The number of trajectories
        num_params : int
            The number of factors
        num_groups : int, default=None
            The number of groups
        local_optimization : bool, default=False
            If True, fills the lower triangle of the distance matrix

        Returns
        -------
        distance_matrix : numpy.ndarray

        """
        if num_groups:
            self.check_input_sample(input_sample, num_groups, num_samples)
        else:
            self.check_input_sample(input_sample, num_params, num_samples)

        index_list = self._make_index_list(num_samples, num_params, num_groups)
        distance_matrix = np.zeros(
            (num_samples, num_samples), dtype=np.float32)

        for j in range(num_samples):
            input_1 = input_sample[index_list[j]]
            for k in range(j + 1, num_samples):
                input_2 = input_sample[index_list[k]]

                # Fills the lower triangle of the matrix
                if local_optimization is True:
                    distance_matrix[j, k] = self.compute_distance(
                        input_1, input_2)

                distance_matrix[k, j] = self.compute_distance(input_1, input_2)
        return distance_matrix
