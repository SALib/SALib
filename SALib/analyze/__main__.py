from sys import exit
import argparse
import sobol, morris, extended_fast

parser = argparse.ArgumentParser(description='Perform sensitivity analysis on model output')

parser.add_argument('-m', '--method', type=str, choices=['sobol', 'morris', 'fast'], required=True)
parser.add_argument('-p', '--paramfile', type=str, required=True, help='Parameter range file')
parser.add_argument('-Y', '--model-output-file', type=str, required=True, help='Model output file')
parser.add_argument('-c', '--column', type=int, required=False, default=0, help='Column of output to analyze')
parser.add_argument('--delimiter', type=str, required=False, default=' ', help='Column delimiter in model output file')
parser.add_argument('--sobol-max-order', type=int, required=False, default=2, choices=[1, 2], help='Maximum order of sensitivity indices to calculate (Sobol only)')
parser.add_argument('-X', '--morris-model-input', type=str, required=False, default=None, help='Model inputs (required for Method of Morris only)')
parser.add_argument('-r', '--sobol-bootstrap-resamples', type=int, required=False, default=1000, help='Number of bootstrap resamples for Sobol confidence intervals')
args = parser.parse_args()

if args.method == 'sobol':
    calc_second_order = (args.sobol_max_order == 2)
    sobol.analyze(args.paramfile, args.model_output_file, args.column, calc_second_order, num_resamples = args.sobol_bootstrap_resamples, delim = args.delimiter)
elif args.method == 'morris':
    if args.morris_model_input is not None:
        morris.analyze(args.paramfile, args.morris_model_input, args.model_output_file, args.column, delim = args.delimiter)
    else:
        print "Error: model input file is required for Method of Morris. Run with -h flag to see usage."
        exit()
elif args.method == 'fast':
    extended_fast.analyze(args.paramfile, args.model_output_file, args.column, delim = args.delimiter)