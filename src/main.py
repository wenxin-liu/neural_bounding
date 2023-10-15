import argparse
import torch

from src.train import train

if __name__ == '__main__':
    torch.manual_seed(0)

    # create the parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--object_name', type=str, help='enter object name for training dataset')

    parser.add_argument(
        '--export_fn',
        type=str,
        help='enter filename prefix for exporting data. '
             'Should contain object name, query type and dimension, nn architecture and class weight')

    parser.add_argument(
        "--query",
        choices=[
            "point", "ray", "plane", "box",
        ],
        default=None,
        help="spatial query type")

    parser.add_argument(
        "--dim",
        type=int,
        choices=[
            2, 3, 4,
        ],
        default=None,
        help="indicator dimension")

    # parse the arguments
    args = parser.parse_args()

    train(object_name="bunny", query="ray", dimension=3)
