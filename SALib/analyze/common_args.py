import argparse


def setup(parser):
    parser = argparse.ArgumentParser(
        description='Perform sensitivity analysis on model output')
    parser.add_argument(
        '-p', '--paramfile', type=str, required=True,
        help='Parameter range file')
    parser.add_argument(
        '-Y', '--model-output-file', type=str, required=True,
        help='Model output file')
    parser.add_argument('-c', '--column', type=int, required=False,
                        default=0, help='Column of output to analyze')
    parser.add_argument('--delimiter', type=str, required=False, default=' ',
                        help='Column delimiter in model output file')
    return parser


def create(cli_parser=None):
    parser = argparse.ArgumentParser(
        description='Perform sensitivity analysis on model output')
    parser = setup(parser)

    if cli_parser:
        parser = cli_parser(parser)

    return parser


def run_cli(cli_parser, run_analysis, known_args=None):
    parser = create(cli_parser)
    args = parser.parse_args(known_args)

    run_analysis(args)
