"""Defines a family of algorithms for generating samples

The sample a for use with :class:`SALib.analyze.morris.analyze`,
encapsulate each one, and makes them interchangeable.

Example
-------
>>> localoptimisation = LocalOptimisation()
>>> context = SampleMorris(localoptimisation)
>>> context.sample(input_sample, num_samples, num_params, k_choices, groups)
"""

import abc
from .. morris_util import compile_output


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

        output = compile_output(input_sample,
                                num_samples,
                                num_params,
                                maximum_combo,
                                num_groups)
        return output

    @abc.abstractmethod
    def _sample(self, input_sample, num_samples,
                num_params, k_choices, groups):
        pass
