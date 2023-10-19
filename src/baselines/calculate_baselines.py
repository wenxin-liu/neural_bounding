from src.baselines.AAElli import calculate_AAElli
from src.baselines.AABox import calculate_AABox
from src.baselines.OElli import calculate_OElli
from src.baselines.helper import extract_ground_truth_classes
from src.baselines.OBox import calculate_OBox
from src.baselines.Sphere import calculate_Sphere
from src.baselines.kDOP import calculate_kDOP
from src.data.data_exporter import DataExporter
from src.metrics.helper import print_metrics
from src.metrics.metrics_registry import MetricsRegistry
from src.wiring import get_source_data, get_training_data


def calculate_baselines(object_name, query, dimension):
    if dimension == 2:
        n_objects = 25_000
    elif dimension == 3:
        n_objects = 62_500
    else:
        n_objects = 100_000

    n_samples = 1500 if dimension == 4 else 500
    dim = dimension if query == 'point' else dimension*2

    data = get_source_data(object_name=object_name, dimension=dimension)
    data_exporter = DataExporter(f'{object_name}_{dimension}d_{query}_query', "aabb")

    metrics_registry = MetricsRegistry()

    features, targets = get_training_data(data=data, query=query, dimension=dimension, n_objects=n_objects,
                                          n_samples=n_samples)

    result = extract_ground_truth_classes(features, targets)
    gt_positive = result["gt_positive"]
    gt_negative = result["gt_negative"]

    print("AABox")
    calculate_AABox(gt_positive=gt_positive, gt_negative=gt_negative, metrics_registry=metrics_registry)
    print_metrics(metrics_registry)

    print("OBox")
    calculate_OBox(gt_positive=gt_positive, gt_negative=gt_negative, metrics_registry=metrics_registry)
    print_metrics(metrics_registry)

    print("Sphere")
    calculate_Sphere(gt_positive=gt_positive, gt_negative=gt_negative, metrics_registry=metrics_registry, dim=dim)
    print_metrics(metrics_registry)

    print("AAElli")
    calculate_AAElli(gt_positive=gt_positive, gt_negative=gt_negative, metrics_registry=metrics_registry, dim=dim)
    print_metrics(metrics_registry)

    print("OElli")
    calculate_OElli(gt_positive=gt_positive, gt_negative=gt_negative, metrics_registry=metrics_registry, dim=dim)
    print_metrics(metrics_registry)

    print("kDOP")
    calculate_kDOP(gt_positive=gt_positive, gt_negative=gt_negative, metrics_registry=metrics_registry, dim=dim)
    print_metrics(metrics_registry)
