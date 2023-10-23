import argparse
from pathlib import Path

import torch

from src.baselines.calculate_baselines import calculate_baselines
from src.data.data_exporter import DataExporter
from src.data.data_importer import import_dataset_from_gdrive
from src.metrics.metrics_registry import MetricsRegistry
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

    # parse the command line arguments
    args = parser.parse_args()

    # import dataset from google drive
    parent_directory = Path(__file__).resolve().parents[1]
    resource_path = parent_directory / 'resources' / f'{args.dim}d'
    import_dataset_from_gdrive(resource_path, dim=args.dim)

    # instantiate metrics registry for storing metrics
    metrics_registry = MetricsRegistry()

    # instantiate data exporter for saving experiment results to file
    data_exporter = DataExporter(object_name=args.object_name, dimension=args.dim, query=args.query)

    # calculate all baseline results
    calculate_baselines(object_name=args.object_name, query=args.query, dimension=args.dim,
                        metrics_registry=metrics_registry)

    # run oursKDOP results
    train_ours_kdop(object_name=args.object_name, query=args.query, dimension=args.dim,
                    metrics_registry=metrics_registry)

    # run oursNeural results
    train_ours_neural(object_name=args.object_name, query=args.query, dimension=args.dim,
                      metrics_registry=metrics_registry)

    # save all results to file in the 'results' directory at the project root
    data_exporter.export_results(metrics_registry)

