from __future__ import division
import numpy as np
import random as rd
from . import common_args
from . sample import Sample
from . morris_util import *


class Morris(Sample):
    '''
    A class which implements three variants of Morris' sampling for
    elementary effects:
            - vanilla Morris
            - optimised trajectories (Campolongo's enhancements from 2007)
            - groups with optimised trajectories (again Campolongo 2007)

    At present, optimised trajectories is implemented using a brute-force
    approach, which can be very slow, especially if you require more than four
    trajectories.  Note that the number of factors makes little difference,
    but the ratio between number of optimal trajectories and the sample size
    results in an exponentially increasing number of scores that must be
    computed to find the optimal combination of trajectories.

    I suggest going no higher than 4 from a pool of 100 samples.

    Suggested enhancements:
        - a parallel brute force method (incomplete)
        - a combinatorial optimisation approach (completed, but dependencies are
          not open-source)
    '''


    def __init__(self, parameter_file, samples, \
                 num_levels, grid_jump, \
                 optimal_trajectories=None):

        Sample.__init__(self, parameter_file, samples)
        self.num_levels = num_levels
        self.grid_jump = grid_jump        
        self.optimal_trajectories = optimal_trajectories

        if self.groups is None:
            sample = self.sample_oat()
        else:
            sample = self.sample_groups()

        if self.optimal_trajectories is not None:
            if self.optimal_trajectories >= self.samples:
                raise ValueError("The number of optimal trajectories should be less than the number of samples.")
            elif self.optimal_trajectories > 10:
                raise ValueError("Running optimal trajectories greater than values of 10 will take a long time.")
            elif self.optimal_trajectories < 2:
                raise ValueError("The number of optimal trajectories must be set to 2 or more.")
            else:
                sample = self.optimize_trajectories(sample)

        self.output_sample = sample


    def sample_oat(self):

        D = self.num_vars
        N = self.samples

        # orientation matrix B: lower triangular (1) + upper triangular (-1)
        B = np.tril(np.ones([D + 1, D], dtype=int), -1) + \
            np.triu(-1 * np.ones([D + 1, D], dtype=int))

        # grid step delta, and final sample matrix X
        delta = self.grid_jump / (self.num_levels - 1)
        X = np.empty([N * (D + 1), D])

        # Create N trajectories. Each trajectory contains D+1 parameter sets.
        # (Starts at a base point, and then changes one parameter at a time)
        for j in range(N):

            # directions matrix DM - diagonal matrix of either +1 or -1
            DM = np.diag([rd.choice([-1, 1]) for _ in range(D)])

            # permutation matrix P
            perm = np.random.permutation(D)
            P = np.zeros([D, D])
            for i in range(D):
                P[i, perm[i]] = 1

            # starting point for this trajectory
            x_base = np.empty([D + 1, D])
            for i in range(D):
                x_base[:, i] = (
                    rd.choice(np.arange(self.num_levels - self.grid_jump))) / (self.num_levels - 1)

            # Indices to be assigned to X, corresponding to this trajectory
            index_list = np.arange(D + 1) + j * (D + 1)
            delta_diag = np.diag([delta for _ in range(D)])

            X[index_list, :] = 0.5 * \
                (np.mat(B) * np.mat(P) * np.mat(DM) + 1) * \
                np.mat(delta_diag) + np.mat(x_base)

        return X

    def sample_groups(self):
        '''
        Returns an N(g+1)-by-k array of N trajectories;
        where g is the number of groups and k is the number of factors
        '''
        N = self.samples
        G = self.groups

        if G is None:
            raise ValueError("Please define the matrix G.")
        if type(G) is not np.matrixlib.defmatrix.matrix:
           raise TypeError("Matrix G should be formatted as a numpy matrix")

        k = G.shape[0]
        g = G.shape[1]
        sample = np.empty((N*(g + 1), k))
        sample = np.array([generate_trajectory(G, self.num_levels, self.grid_jump) for n in range(N)])
        return sample.reshape((N*(g + 1), k))


    def optimize_trajectories(self, input_sample):

        N = self.samples
        k_choices = self.optimal_trajectories

        if np.any((input_sample < 0) | (input_sample > 1)):
            raise ValueError("Input sample must be scaled between 0 and 1")

        scores = find_most_distant(input_sample, N, self.num_vars, k_choices)

        index_list = []
        for j in range(N):
            index_list.append(np.arange(self.num_vars + 1) + j * (self.num_vars + 1))

        maximum_combo = find_maximum(scores, N, k_choices)
        output = np.zeros((np.size(maximum_combo) * (self.num_vars + 1), self.num_vars))
        for counter, x in enumerate(maximum_combo):
            output[index_list[counter]] = np.array(input_sample[index_list[x]])
        return output


    def debug(self):
        print("Parameter File: %s" % self.parameter_file)
        print("Number of samples: %s" % self.samples)
        print("Number of levels: %s" % self.num_levels)
        print("Grid step: %s" % self.grid_jump)
        print("Number of variables: %s" % self.num_vars)
        print("Parameter bounds: %s" % self.bounds)
        if self.groups is not None:
          print("Group: %s" % self.groups)
        if self.optimal_trajectories is not None:
          print("Number of req trajectories: %s" % self.optimal_trajectories)


if __name__ == "__main__":

    parser = common_args.create()
    parser.add_argument('-l','--levels', type=int, required=False,
                        default=4, help='Number of grid levels (Morris only)')
    parser.add_argument('--grid-jump', type=int, required=False,
                        default=2, help='Grid jump size (Morris only)')
    parser.add_argument('-k','--k-optimal', type=int, required=False,
                        default=None, help='Number of optimal trajectories (Morris only)')
    args = parser.parse_args()

    np.random.seed(args.seed)
    rd.seed(args.seed)

    sample = Morris(args.paramfile, args.samples, args.levels, \
                    args.grid_jump, args.group, args.k_optimal)

    sample.save_data(args.output, delimiter=args.delimiter, precision=args.precision)
