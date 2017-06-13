"""Defines a family of algorithms for generating samples

The sample a for use with :class:`SALib.analyze.morris.analyze`,
encapsulate each one, and makes them interchangeable.

Example
-------
>>> localoptimisation = LocalOptimisation()
>>> context = SampleMorris(localoptimisation)
>>> context.sample(input_sample, N, num_params, k_choices, groups)
"""

import abc
from SALib.sample.morris_util import find_locally_optimum_combination


class SampleMorris:
    """
    Define the interface of interest to clients.
    Maintain a reference to a Strategy object.
    """

    def __init__(self, strategy):
        self._strategy = strategy

    def sample(self, input_sample, N, num_params, k_choices, groups):
        self._strategy.sample(input_sample, N, num_params, k_choices, groups)


class Strategy(metaclass=abc.ABCMeta):
    """
    Declare an interface common to all supported algorithms. Context
    uses this interface to call the algorithm defined by a
    ConcreteStrategy.
    """

    @abc.abstractmethod
    def sample(self, input_sample, N, num_params, k_choices, groups):
        pass


class LocalOptimisation(Strategy):
    """
    Implement the algorithm using the Strategy interface.
    """

    def sample(self, input_sample, N, num_params, k_choices, groups):
        return find_locally_optimum_combination(
            input_sample, N, num_params, k_choices, groups)


class GlobalOptimisation(Strategy):
    """
    Implement the algorithm using the Strategy interface.
    """

    def sample(self, input_sample, N, num_params, k_choices, groups):
        pass


class BruteForce(Strategy):

    def sample(self, input_sample, N, num_params, k_choices, groups):
        pass


def main():
    pass


if __name__ == "__main__":
    main()
