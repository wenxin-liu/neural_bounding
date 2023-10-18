from src.baselines.aabb import calculate_aabb
from src.data.data_exporter import DataExporter
from src.metrics.metrics_registry import MetricsRegistry
from src.wiring import get_source_data, get_training_data


def calculate_baselines(object_name, query, dimension):
    if query == 'point' and dimension == 2:
        n_objects = 25_000
    elif query != 'point' and dimension == 2:
        n_objects = 250_000
    elif query == 'point' and dimension == 3:
        n_objects = 62_500
    elif query != 'point' and dimension == 3:
        n_objects = 62500
    elif query == 'point' and dimension == 4:
        n_objects = 100_000
    else:
        n_objects = 100_000

    n_samples = 1500 if dimension == 4 else 500

    data = get_source_data(object_name=object_name, dimension=dimension)
    data_exporter = DataExporter(f'{object_name}_{dimension}d_{query}_query', "aabb")

    metrics_registry = MetricsRegistry()

    features, targets = get_training_data(data=data, query=query, dimension=dimension, n_objects=n_objects,
                                          n_samples=n_samples)

    calculate_aabb(features, targets, metrics_registry)

    metrics = metrics_registry.get_metrics()
    for key, value in metrics.items():
        print(f"{key}: {value}")
