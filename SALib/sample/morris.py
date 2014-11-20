from __future__ import division
import numpy as np
import random as rd
from . import common_args
from ..sample import morris_oat, morris_groups, morris_optimal
from ..util import read_param_file, scale_samples


class Sample(object):
    '''
    A template class, which all of the sample classes inherit.
    '''

    def __init__(self, parameter_file, samples):

        self.parameter_file = parameter_file
        pf = read_param_file(self.parameter_file)
        self.num_vars = pf['num_vars']
        self.bounds = pf['bounds']
        self.samples = samples
        self.output_sample = None


    def save_data(self, output, delimiter, precision):
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
        return scale_samples(self.output_sample, self.bounds)


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
    but the ratio between number of factors and the sample size results in an
    exponentially increasing number of scores that must be computed to find
    the optimal combination of trajectories.

    I suggest going no higher than 4 from a pool of 100 samples.

    Suggested enhancements:
        - a parallel brute force method (incomplete)
        - a combinatorial optimisation approach (completed, but dependencies are
          not open-source)
    '''


    def __init__(self, parameter_file, samples, num_levels, grid_jump, \
                 group=None, optimal_trajectories=None):

        self.parameter_file = parameter_file
        self.samples = samples
        self.num_levels = num_levels
        self.grid_jump = grid_jump
        pf = read_param_file(self.parameter_file)
        self.num_vars = pf['num_vars']
        self.bounds = pf['bounds']
        self.group = group
        self.optimal_trajectories = optimal_trajectories

        if self.optimal_trajectories != None:
            # Check to ensure that fewer optimal trajectories than samples are
            # requested, otherwise ignore
            if self.optimal_trajectories >= self.samples:
                raise ValueError("The number of optimal trajectories should be less than the number of samples.")
            elif self.optimal_trajectories > 4:
                raise ValueError("Running optimal trajectories greater than values of 10 will take a long time.")
            elif self.optimal_trajectories <= 1:
                raise ValueError("The number of optimal trajectories must be set to 2 or more.")

        if self.group is None:

            self.create_sample()

        else:

            self.create_sample_with_groups()




    def create_sample(self):

        if self.optimal_trajectories is None:

            optimal_sample = morris_oat.sample(self.samples,
                                               self.parameter_file,
                                               self.num_levels,
                                               self.grid_jump)

        else:

            sample = morris_oat.sample(self.samples,
                                       self.parameter_file,
                                       self.num_levels,
                                       self.grid_jump)
            optimal_sample = \
                morris_optimal.find_optimum_trajectories(sample,
                                                         self.samples,
                                                         self.num_vars,
                                                         self.optimal_trajectories)

        self.output_sample = optimal_sample


    def create_sample_with_groups(self):
        self.output_sample = morris_groups.sample(self.samples,
                                                  self.group,
                                                  self.num_levels,
                                                  self.grid_jump)
        if self.optimal_trajectories is not None:
            self.output_sample = \
                morris_optimal.find_optimum_trajectories(self.output_sample,
                                                         self.samples,
                                                         self.num_vars,
                                                         self.optimal_trajectories)
        scale_samples(self.output_sample, self.bounds)


    def debug(self):
        print "Parameter File: %s" % self.parameter_file
        print "Number of samples: %s" % self.samples
        print "Number of levels: %s" % self.num_levels
        print "Grid step: %s" % self.grid_jump
        print "Number of variables: %s" % self.num_vars
        print "Parameter bounds: %s" % self.bounds
        print "Group: %s" % self.group
        print "Number of req trajectories: %s" % self.optimal_trajectories


if __name__ == "__main__":

    parser = common_args.create()
    parser.add_argument('-l','--levels', type=int, required=False,
                        default=4, help='Number of grid levels (Morris only)')
    parser.add_argument('--grid-jump', type=int, required=False,
                        default=2, help='Grid jump size (Morris only)')
    parser.add_argument('-k','--k-optimal', type=int, required=False,
                        default=None, help='Number of optimal trajectories (Morris only)')
    parser.add_argument('--group', type=str, required=False, default=None,
                       help='File path to grouping file (Morris only)')
    args = parser.parse_args()

    np.random.seed(args.seed)
    rd.seed(args.seed)

    sample = Morris(args.paramfile, args.samples, args.levels, \
                    args.grid_jump, args.group, args.k_optimal)

    sample.save_data(args.output, delimiter=args.delimiter, precision=args.precision)
