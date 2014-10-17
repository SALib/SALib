import argparse


def create():
    parser = argparse.ArgumentParser(
        description='Create parameter samples for sensitivity analysis')

    parser = argparse.ArgumentParser(
        description='Perform sensitivity analysis on model output')
    parser.add_argument(
        '-p', '--paramfile', type=str, required=True, help='Parameter range file')
    parser.add_argument(
        '-Y', '--model-output-file', type=str, required=True, help='Model output file')
    parser.add_argument('-c', '--column', type=int, required=False,
                        default=0, help='Column of output to analyze')
    parser.add_argument('--delimiter', type=str, required=False,
                        default=' ', help='Column delimiter in model output file')

    return parser
