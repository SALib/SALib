import argparse


def setup(parser):
    """Add common sampling options to CLI parser.

    Parameters
    ----------
    parser : argparse object

    Returns
    ----------
    Updated argparse object
    """
    parser.add_argument('-n', '--samples', type=int, required=True,
                        help='Number of Samples')
    parser.add_argument('-p', '--paramfile', type=str, required=True,
                        help='Parameter Range File')
    parser.add_argument('-o', '--output', type=str, required=True, 
                        help='Output File')
    parser.add_argument('-s', '--seed', type=int, required=False, 
                        default=None,
                        help='Random Seed')
    parser.add_argument('--delimiter', type=str, required=False,
                        default=' ',
                        help='Column delimiter')
    parser.add_argument('--precision', type=int, required=False,
                        default=8,
                        help='Output floating-point precision')

    return parser


def create(cli_parser=None):
    """Create CLI parser object.

    Parameters
    ----------
    cli_parser : function [optional]
        Function to add method specific arguments to parser

    Returns
    ----------
    argparse object
    """
    parser = argparse.ArgumentParser(
        description='Create parameter samples for sensitivity analysis')
    parser = setup(parser)

    if cli_parser:
        parser = cli_parser(parser)

    return parser


def run_cli(cli_parser, run_sample, known_args=None):
    """Run sampling with CLI arguments.

    Parameters
    ----------
    cli_parser : function
        Function to add method specific arguments to parser
    run_sample: function
        Method specific function that runs the sampling
    known_args: list [optional]
        Additional arguments to parse

    Returns
    ----------
    argparse object
    """
    parser = create(cli_parser)
    args = parser.parse_args(known_args)

    run_sample(args)
