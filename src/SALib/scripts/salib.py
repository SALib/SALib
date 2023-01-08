"""Command-line utility for SALib"""
import importlib
import argparse
from SALib import analyze, sample
from SALib.util import avail_approaches


def parse_subargs(module, parser, method, opts):
    """Attach argument parser for action specific options.

    Parameters
    ----------
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
    -------
    subargs : argparser namespace object
    """
    module.cli_args(parser)
    subargs = parser.parse_args(opts)
    return subargs


def main():
    parser = argparse.ArgumentParser(description="SALib - Sensitivity Analysis Library")
    subparsers = parser.add_subparsers(help="Sample or Analysis method", dest="action")
    sample_parser = subparsers.add_parser(
        "sample", description="Select one of the sample methods"
    )
    analyze_parser = subparsers.add_parser("analyze")

    # Get list of available samplers and analyzers
    samplers = avail_approaches(sample)
    analyzers = avail_approaches(analyze)
    sample_parser.add_argument("method", choices=samplers)
    analyze_parser.add_argument("method", choices=analyzers)

    known_args, opts = parser.parse_known_args()

    action = known_args.action
    if not action:
        parser.print_help()
        exit()

    method_name = known_args.method
    common_args = importlib.import_module(".".join(["SALib", action, "common_args"]))
    module = importlib.import_module(".".join(["SALib", action, method_name]))

    if len(opts) == 0:
        cmd_parse = common_args.create(module.cli_parse)
        cmd_parse.print_help()
        exit()

    common_args.run_cli(module.cli_parse, module.cli_action, opts)


if __name__ == "__main__":
    main()
