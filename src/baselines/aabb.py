import torch

from src.baselines.helper import extract_ground_truth_classes


def calculate_aabb(features, targets, metrics_registry):
    # clean metrics registry
    metrics_registry.reset_metrics()

    result = extract_ground_truth_classes(features, targets)
    gt_positive = result["gt_positive"]
    gt_negative = result["gt_negative"]

    # find aabb min and max coordinates
    aabb_min = torch.min(gt_positive, dim=0)[0]
    aabb_max = torch.max(gt_positive, dim=0)[0]

    # take all the true negative points and calculate which fall inside the aabb
    is_inside_aabb = torch.all(aabb_min <= gt_negative, dim=1) & torch.all(gt_negative <= aabb_max, dim=1)

    # this produces the false positive values for aabb
    fp_aabb = torch.sum(is_inside_aabb).item()

    # take all the true positive points and calculate which fall outside the AABB
    is_outside_aabb = torch.any(aabb_min > gt_positive, dim=1) | torch.any(gt_positive > aabb_max, dim=1)

    # this produces the false negative values for aabb - should be 0
    fn_aabb = torch.sum(is_outside_aabb).item()

    metrics_registry.register_counter_metric("false_negative")
    metrics_registry.register_counter_metric("false_positive")
    metrics_registry.register_counter_metric("true_value")
    metrics_registry.register_counter_metric("total_samples")

    metrics_registry.add("false_negative", fn_aabb)
    metrics_registry.add("false_positive", fp_aabb)
    metrics_registry.add("true_value", gt_positive.shape[0] + gt_negative.shape[0] - fp_aabb - fn_aabb)
    metrics_registry.add("total_samples", gt_positive.shape[0] + gt_negative.shape[0])
