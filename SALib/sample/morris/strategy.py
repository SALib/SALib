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
from itertools import combinations, islice


class SampleMorris(object):
    """Computes the optimum `k_choices` of trajectories from the input_sample.

    Arguments
    ---------
    strategy : :class:`Strategy`
    """

    def __init__(self, strategy):
        self._strategy = strategy

    def sample(self, input_sample, num_samples, num_params, k_choices, groups):
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
        groups : tuple
            A tuple of (numpy.ndarray, list)
        """
        return self._strategy.sample(input_sample, num_samples, num_params,
                                     k_choices, groups)


class Strategy(metaclass=abc.ABCMeta):
    """
    Declare an interface common to all supported algorithms.
    :class:`SampleMorris` uses this interface to call the algorithm
    defined by a ConcreteStrategy.
    """

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

    def sample(self, input_sample, num_samples, num_params,
               k_choices, groups):
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
        groups : tuple
            A tuple of (numpy.ndarray, list)
        """
        self.run_checks(num_samples, k_choices)
        maximum_combo = self._sample(input_sample, num_samples,
                                     num_params, k_choices, groups)

        num_groups = None
        if groups is not None:
            num_groups = groups[0].shape[1]

        output = self.compile_output(input_sample,
                                     num_samples,
                                     num_params,
                                     maximum_combo,
                                     num_groups)
        return output

    @abc.abstractmethod
    def _sample(self, input_sample, num_samples,
                num_params, k_choices, groups):
        """Implement this in your class
        """
        pass

    @staticmethod
    def _make_index_list(num_samples, num_params, groups=None):
        """

        For each sample, appends the array of indices

        Returns
        -------
        list of numpy.ndarray
        """
        if groups is None:
            groups = num_params

        index_list = []
        for j in range(num_samples):
            index_list.append(np.arange(groups + 1) + j * (groups + 1))
        return index_list

    def compile_output(self, input_sample, num_samples, num_params,
                       maximum_combo, groups=None):
        """
        """

        if groups is None:
            groups = num_params

        self.check_input_sample(input_sample, groups, num_samples)

        index_list = self._make_index_list(num_samples, num_params, groups)

        output = np.zeros((np.size(maximum_combo) * (groups + 1), num_params))
        for counter, x in enumerate(maximum_combo):
            output[index_list[counter]] = np.array(input_sample[index_list[x]])
        return output

    @staticmethod
    def check_input_sample(input_sample, num_params, num_samples):
        '''
        Checks input sample is:
            - the correct size
            - values between 0 and 1
        '''
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
            print("Trajectory %s and %s are equal" % (m, l))
            distance = 0
        else:
            distance = np.array(np.sum(cdist(m, l)), dtype=np.float32)

        return distance

    def compute_distance_matrix(self, input_sample, num_samples, num_params,
                                groups=None,
                                local_optimization=False):
        """Computes the distance between every trajectory

        Each entry in the matrix represents the sum of the geometric distances
        between all the couples of points of the two fixed trajectories

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
        groups : tuple, default=None
            the mapping between factors and groups
        local_optimization : bool, default=False
            If True, fills the lower triangle of the distance matrix

        Returns
        -------
        distance_matrix : numpy.ndarray

        """
        num_groups = None
        if groups:
            num_groups = groups[0].shape[1]
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

    def find_maximum(self, scores, N, k_choices):
        """Finds the `k_choices` maximum scores from `scores`

        Arguments
        ---------
        scores : numpy.ndarray
        N : int
        k_choices : int
        """
        if not isinstance(scores, np.ndarray):
            raise TypeError("Scores input is not a numpy array")

        index_of_maximum = int(scores.argmax())
        maximum_combo = self.nth(combinations(
            list(range(N)), k_choices), index_of_maximum, None)
        return sorted(maximum_combo)

    @staticmethod
    def nth(iterable, n, default=None):
        """Returns the nth item or a default value

        Arguments
        ---------
        iterable : iterable
        n : int
        default : default=None
            The default value to return
        """

        if type(n) != int:
            raise TypeError("n is not an integer")

        return next(islice(iterable, n, None), default)
