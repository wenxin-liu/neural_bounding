import argparse
import torch

from src.ours_kdop.train_ours_kdop import train_ours_kdop
from src.ours_neural.train_ours_neural import train_ours_neural

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

    print(f"oursKDOP {args.object_name}, {args.query}, {args.dim}")
    train_ours_kdop(object_name=args.object_name, query=args.query, dimension=args.dim)

    print(f"oursNeural {args.object_name}, {args.query}, {args.dim}")
    train_ours_neural(object_name=args.object_name, query=args.query, dimension=args.dim)
