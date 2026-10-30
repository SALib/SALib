import argparse

import numpy as np


def setup(parser):
    parser = argparse.ArgumentParser(
        description="Perform sensitivity analysis on model output"
    )
    parser.add_argument(
        "-p", "--paramfile", type=str, required=True, help="Parameter range file"
    )
    parser.add_argument(
        "-Y", "--model-output-file", type=str, required=True, help="Model output file"
    )
    parser.add_argument(
        "-c",
        "--column",
        type=int,
        required=False,
        default=0,
        help="Column of output to analyze",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        required=False,
        default=" ",
        help="Column delimiter in model output file",
    )
    parser.add_argument(
        "-s", "--seed", type=int, required=False, default=None, help="Random Seed"
    )
    return parser


def create(cli_parser=None):
    parser = argparse.ArgumentParser(
        description="Perform sensitivity analysis on model output"
    )
    parser = setup(parser)

    if cli_parser:
        parser = cli_parser(parser)

    return parser


def run_cli(cli_parser, run_analysis, known_args=None):
    parser = create(cli_parser)
    args = parser.parse_args(known_args)

    run_analysis(args)


def load_inputs(args, problem, check_num_vars=True):
    """Load the model inputs X and outputs Y named on the command line.

    Applies the checks `ProblemSpec` makes on samples and results: X must
    have one column per parameter and as many rows as Y. FF samples carry
    dummy columns, so `ff` passes `check_num_vars=False`.
    """
    X = np.loadtxt(args.model_input_file, delimiter=args.delimiter, ndmin=2)
    Y = np.loadtxt(
        args.model_output_file,
        delimiter=args.delimiter,
        usecols=(args.column,),
        ndmin=1,
    )

    if check_num_vars and X.shape[1] != problem["num_vars"]:
        raise ValueError(
            f"Mismatched sample size: Expected {problem['num_vars']} cols "
            f"in {args.model_input_file}, got {X.shape[1]}"
        )
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"Mismatched number of rows: {args.model_input_file} has "
            f"{X.shape[0]}, {args.model_output_file} has {Y.shape[0]}"
        )

    return X, Y
