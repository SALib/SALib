"""
"""
from SALib.sample.morris_strategies import Strategy
from .. morris_util import (find_most_distant, find_maximum)


class BruteForce(Strategy):
    """Implements the brute force optimisation strategy
    """

    def _sample(self, input_sample, num_samples,
                num_params, k_choices, groups):
        return self.brute_force_most_distant(input_sample, num_samples,
                                             num_params, k_choices, groups)

    def brute_force_most_distant(self, input_sample, N, num_params, k_choices,
                                 groups=None):
        """Use brute force method to find most distant trajectories

        Arguments
        ---------
        input_sample
        N : int
        num_params : int
        k_choices : int
        groups : default=None,
        """
        scores = find_most_distant(input_sample,
                                   N,
                                   num_params,
                                   k_choices,
                                   groups)

        maximum_combo = find_maximum(scores, N, k_choices)

        return maximum_combo
