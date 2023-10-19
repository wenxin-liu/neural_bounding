import torch


def calculate_obb(gt_positive, gt_negative, metrics_registry):
    # clean metrics registry
    metrics_registry.reset_metrics()

    covariance = torch.cov(gt_positive.T)

    eigen_values, eigen_vectors = torch.linalg.eig(covariance)
    eigen_values = eigen_values.real
    eigen_vectors = eigen_vectors.real

    gt_positive_transformed = gt_positive @ eigen_vectors
    gt_negative_transformed = gt_negative @ eigen_vectors

    # find obb min and max coordinates
    obb_min = torch.min(gt_positive_transformed, dim=0)[0]
    obb_max = torch.max(gt_positive_transformed, dim=0)[0]

    # take all the true negative points and calculate which fall inside the obb
    is_inside_obb = torch.all(obb_min <= gt_negative_transformed, dim=1) & torch.all(gt_negative_transformed <= obb_max,
                                                                                     dim=1)

    # this produces the false positive values for obb
    fp_obb = torch.sum(is_inside_obb).item()

    # take all the true positive points and calculate which fall outside the obb
    is_outside_obb = torch.any(obb_min > gt_positive_transformed, dim=1) | torch.any(gt_positive_transformed > obb_max,
                                                                                     dim=1)

    # this produces the false negative values for obb - should be 0
    fn_obb = torch.sum(is_outside_obb).item()

    metrics_registry.register_counter_metric("false_negative")
    metrics_registry.register_counter_metric("false_positive")
    metrics_registry.register_counter_metric("true_value")
    metrics_registry.register_counter_metric("total_samples")

    metrics_registry.add("false_negative", fn_obb)
    metrics_registry.add("false_positive", fp_obb)
    metrics_registry.add("true_value", gt_positive.shape[0] + gt_negative.shape[0] - fp_obb - fn_obb)
    metrics_registry.add("total_samples", gt_positive.shape[0] + gt_negative.shape[0])
