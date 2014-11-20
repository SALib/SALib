from __future__ import division
import numpy as np
import random as rd
from . import common_args
from ..sample import morris_oat, morris_groups, morris_optimal
from ..util import read_param_file

class Sample(object):


    def __init__(self, parameter_file, samples):

        self.parameter_file = parameter_file
        self.samples = samples
        self.output_sample = None


    def save_data(self, output, delimiter, precision):

        np.savetxt(output,
                   self.output_sample,
                   delimiter=delimiter,
                   fmt='%.' + str(precision) + 'e'
                   )


class Morris(Sample):


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

        self.debug()

        if self.optimal_trajectories != None:
            # Check to ensure that fewer optimal trajectories than samples are
            # requested, otherwise ignore
            if self.optimal_trajectories >= self.samples:
                raise ValueError("The number of optimal trajectories \
                                  should be less than the number of samples.")
            elif self.optimal_trajectores >= 4:
                raise ValueError("Running optimal trajectories greater than  \
                                  values of 4 can take a long time.")

        if self.group == None:

            self.create_sample()

        else:

            self.create_sample_with_groups()


    def create_sample(self):
        self.output_sample = morris_oat.sample(self.samples,
                                               self.parameter_file,
                                               self.num_levels,
                                               self.grid_jump)
        if self.optimal_trajectories:
            self.output_sample = \
                morris_optimal.find_optimum_trajectories(self.output_sample,
                                                         self.samples,
                                                         self.num_vars,
                                                         self.optimal_trajectories
                                                         )


    def create_sample_with_groups(self):
        self.output_sample = morris_groups.sample(self.samples,
                                                  self.group,
                                                  self.parameter_file,
                                                  self.num_levels,
                                                  self.grid_jump)
        if self.optimal_trajectories:
            self.output_sample = \
                morris_optimal.find_optimum_trajectories(self.output_sample,
                                                         self.samples,
                                                         self.num_vars,
                                                         self.optimal_trajectories
                                                         )


    def debug(self):
        print self.parameter_file
        print self.samples
        print self.num_levels
        print self.grid_jump
        print self.num_vars
        print self.bounds
        print self.group
        print self.optimal_trajectories


if __name__ == "__main__":

    parser = common_args.create()
    parser.add_argument('--num-levels', type=int, required=False,
                        default=4, help='Number of grid levels (Morris only)')
    parser.add_argument('--grid-jump', type=int, required=False,
                        default=2, help='Grid jump size (Morris only)')
    parser.add_argument('--opt', type=int, required=False,
                        default=4, help='Number of optimal trajectories (Morris only)')
    parser.add_argument('--group', type=str, required=False,
                       help='File path to grouping file (Morris only)')
    args = parser.parse_args()

    np.random.seed(args.seed)
    rd.seed(args.seed)

    sample = Morris(args.paramfile, args.samples, args.num_levels, \
                    args.grid_jump, args.opt, args.group)
    sample.debug()
    #sample.create_sample()

    #sample.save_data(args.output, delimiter=args.delimiter, precision=args.precision)
