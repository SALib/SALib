import argparse


def create():
    parser = argparse.ArgumentParser(
        description='Create parameter samples for sensitivity analysis')

    parser.add_argument(
        '-p', '--paramfile', type=str, required=True,
        help='Parameter Range File')
    parser.add_argument(
        '-o', '--output', type=str, required=True, help='Output File')
    parser.add_argument(
        '-s', '--seed', type=int, required=False, default=None,
        help='Random Seed')
    parser.add_argument(
        '--delimiter', type=str, required=False, default=' ',
        help='Column delimiter')
    parser.add_argument('--precision', type=int, required=False,
                        default=8, help='Output floating-point precision')
    return parser
