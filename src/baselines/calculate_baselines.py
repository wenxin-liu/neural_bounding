from src.baselines.AAElli import calculate_AAElli
from src.baselines.AABox import calculate_AABox
from src.baselines.OElli import calculate_OElli
from src.baselines.helper import extract_ground_truth_classes
from src.baselines.OBox import calculate_OBox
from src.baselines.Sphere import calculate_Sphere
from src.baselines.kDOP import calculate_kDOP
from src.metrics.helper import print_metrics
from src.wiring import get_source_data, get_training_data


def calculate_baselines(object_name, query, dimension, metrics_registry):
    n_regions = 50_000
    n_samples = 1500 if dimension == 4 else 500

    dim = dimension if query == 'point' else dimension*2

    data = get_source_data(object_name=object_name, dimension=dimension)

    features, targets = get_training_data(data=data, query=query, dimension=dimension, n_regions=n_regions,
                                          n_samples=n_samples)

    result = extract_ground_truth_classes(features, targets)
    gt_positive = result["gt_positive"]
    gt_negative = result["gt_negative"]

    print(f"AABox {object_name} {dimension}D {query} query")
    calculate_AABox(gt_positive=gt_positive, gt_negative=gt_negative, metrics_registry=metrics_registry)
    print_metrics(metrics_registry.metrics_registry["AABox"])

    print(f"OBox {object_name} {dimension}D {query} query")
    calculate_OBox(gt_positive=gt_positive, gt_negative=gt_negative, metrics_registry=metrics_registry)
    print_metrics(metrics_registry.metrics_registry["OBox"])

    print(f"Sphere {object_name} {dimension}D {query} query")
    calculate_Sphere(gt_positive=gt_positive, gt_negative=gt_negative, metrics_registry=metrics_registry, dim=dim)
    print_metrics(metrics_registry.metrics_registry["Sphere"])

    print(f"AAElli {object_name} {dimension}D {query} query")
    calculate_AAElli(gt_positive=gt_positive, gt_negative=gt_negative, metrics_registry=metrics_registry, dim=dim)
    print_metrics(metrics_registry.metrics_registry["AAElli"])

    print(f"OElli {object_name} {dimension}D {query} query")
    calculate_OElli(gt_positive=gt_positive, gt_negative=gt_negative, metrics_registry=metrics_registry, dim=dim)
    print_metrics(metrics_registry.metrics_registry["OElli"])

    print(f"kDOP {object_name} {dimension}D {query} query")
    calculate_kDOP(gt_positive=gt_positive, gt_negative=gt_negative, metrics_registry=metrics_registry, dim=dim)
    print_metrics(metrics_registry.metrics_registry["kDOP"])
