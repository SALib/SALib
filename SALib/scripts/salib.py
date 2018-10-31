'''Command-line utility for SALib'''
import importlib
import argparse
import pkgutil
from SALib import (analyze, sample)


def avail_approaches(pkg):
    '''Create list of available modules.

    Arguments
    ---------
    pkg : module
        name of module to inspect

    Returns
    ---------
    method : list
        A list of available submodules
    '''
    methods = [modname for importer, modname, ispkg in
               pkgutil.walk_packages(path=pkg.__path__)
               if modname != 'common_args']
    return methods


def parse_subargs(module, parser, method, opts):
    '''Attach argument parser for action specific options.

    Arguments
    ---------
    module : module
        name of module to extract action from
    parser : argparser
        argparser object to attach additional arguments to
    method : str
        name of method (morris, sobol, etc).
        Must match one of the available submodules
    opts : list
        A list of argument options to parse

    Returns
    ---------
    subargs : argparser namespace object
    '''
    module.cli_args(parser)
    subargs = parser.parse_args([method] + opts)
    return subargs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SALib - Sensitivity Analysis Library')
    subparsers = parser.add_subparsers(help='Sample or Analysis method',
                                       dest='action')
    sample_parser = subparsers.add_parser('sample')
    analyze_parser = subparsers.add_parser('analyze')

    # Get list of available samplers and analyzers
    samplers = avail_approaches(sample)
    analyzers = avail_approaches(analyze)
    sample_parser.add_argument('method', choices=samplers)
    analyze_parser.add_argument('method', choices=analyzers)

    known_args, opts = parser.parse_known_args()

    action = known_args.action
    if not action:
        parser.print_help()
        import sys
        sys.exit()

    method_name = known_args.method
    module = importlib.import_module('.'.join(['SALib', action, method_name]))
    if action == 'sample':
        parse_obj = sample_parser
        run_func = module.run_sample
    elif action == 'analyze':
        parse_obj = analyze_parser
        run_func = module.run_analysis

    subargs = parse_subargs(module, parse_obj, method_name, opts)
    run_func(subargs)
