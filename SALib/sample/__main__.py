import numpy as np
import random as rd
import argparse
from ..util import read_param_file, scale_samples
import uniform, normal, latin_hypercube, saltelli, morris_oat, fast_sampler

parser = argparse.ArgumentParser(description='Create parameter samples for sensitivity analysis')

parser.add_argument('-m', '--method', type=str, choices=['uniform', 'normal', 'latin', 'saltelli', 'morris', 'fast'], required=True)
parser.add_argument('-n', '--samples', type=int, required=True, help='Number of Samples')
parser.add_argument('-p', '--paramfile', type=str, required=True, help='Parameter Range File')
parser.add_argument('-o', '--output', type=str, required=True, help='Output File')
parser.add_argument('-s', '--seed', type=int, required=False, default=1, help='Random Seed')
parser.add_argument('--delimiter', type=str, required=False, default=' ', help='Column delimiter')
parser.add_argument('--precision', type=int, required=False, default=8, help='Output floating-point precision')
parser.add_argument('--saltelli-max-order', type=int, required=False, default=2, choices=[1, 2], help='Maximum order of sensitivity indices to calculate (Saltelli only)')
parser.add_argument('--morris-num-levels', type=int, required=False, default=10, help='Number of grid levels (Morris only)')
parser.add_argument('--morris-grid-jump', type=int, required=False, default=5, help='Grid jump size (Morris only)')
args = parser.parse_args()


np.random.seed(args.seed)
rd.seed(args.seed)

pf = read_param_file(args.paramfile)

if args.method == 'uniform':
    param_values = uniform.sample(args.samples, pf['num_vars'])
elif args.method == 'normal':
    param_values = normal.sample(args.samples, pf['num_vars'])
elif args.method == 'latin':
    param_values = latin_hypercube.sample(args.samples, pf['num_vars'])
elif args.method == 'saltelli':
    if args.saltelli_max_order == 2:
        param_values = saltelli.sample(args.samples, pf['num_vars'], calc_second_order = True)
    else:
        param_values = saltelli.sample(args.samples, pf['num_vars'], calc_second_order = False)
elif args.method == 'morris':
    param_values = morris_oat.sample(args.samples, pf['num_vars'])
elif args.method == 'fast':
    param_values = fast_sampler.sample(args.samples, pf['num_vars'])

scale_samples(param_values, pf['bounds'])

# Note: for tab-delimited file, enter $'\t' as the delimiter argument on the command line
# Otherwise Bash might mess up the escape characters
np.savetxt(args.output, param_values, delimiter=args.delimiter, fmt='%.' + str(args.precision) + 'e')