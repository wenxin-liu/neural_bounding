import torch


# kDOP implementation - k=4m, where m is the query dimensionality
def calculate_4kDOP_normals(dim):
    if dim < 2:
        raise ValueError("dimensionality must be greater or equals to 2")

    normals = torch.empty(dim * 2, dim)

    for i in range(dim):
        normal = torch.full((dim,), 0., dtype=torch.float)
        normal[i] = 1.
        normals[i] = normal

    for j in range(dim):
        normal = torch.full((dim,), 1., dtype=torch.float)
        normal[j] = 0.
        normals[dim + j] = normal

    return normals


def calculate_kDOP(gt_positive, gt_negative, metrics_registry, dim):
    # clean metrics registry
    metrics_registry.reset_metrics()

    mins = torch.empty(dim * 2, 1)
    maxs = torch.empty(dim * 2, 1)

    normals = calculate_4kDOP_normals(dim)

    for i in range(dim*2):
        new_points = gt_positive @ normals[i].unsqueeze(0).T
        min = new_points.min(axis=0).values
        max = new_points.max(axis=0).values

        mins[i] = min
        maxs[i] = max

    # Initialize a tensor to keep track of cumulative conditions
    cumulative_conditions_fp = torch.full((gt_negative.shape[0],), True, dtype=torch.bool)

    for i in range(dim*2):
        new_points = gt_negative @ normals[i].unsqueeze(0).T
        min_val = mins[i]
        max_val = maxs[i]
        min_condition = torch.all(new_points >= min_val, axis=1)
        max_condition = torch.all(new_points <= max_val, axis=1)

        # use logical AND to make sure both conditions are met
        both_conditions_met = torch.logical_and(min_condition, max_condition)

        # Update the cumulative conditions
        cumulative_conditions_fp = torch.logical_and(cumulative_conditions_fp, both_conditions_met)

    # Count the number of True values in cumulative_conditions
    false_positive = torch.sum(cumulative_conditions_fp).item()

    # Initialise a tensor to keep track of cumulative conditions for false_negative
    cumulative_conditions_fn = torch.full((gt_positive.shape[0],), False, dtype=torch.bool)

    for i in range(dim*2):
        new_points = gt_positive @ normals[i].unsqueeze(0).T
        min_val = mins[i]
        max_val = maxs[i]
        min_condition = torch.any(new_points < min_val, axis=1)
        max_condition = torch.any(new_points > max_val, axis=1)

        # Use logical OR to make sure either condition is met
        both_conditions_met = torch.logical_or(min_condition, max_condition)

        # Update the cumulative conditions for false_negative
        cumulative_conditions_fn = torch.logical_or(cumulative_conditions_fn, both_conditions_met)

    # Count the number of True values in cumulative_conditions_fn
    false_negative = torch.sum(cumulative_conditions_fn).item()

    metrics_registry.register_counter_metric("false_negative")
    metrics_registry.register_counter_metric("false_positive")
    metrics_registry.register_counter_metric("true_value")
    metrics_registry.register_counter_metric("total_samples")

    metrics_registry.add("false_negative", false_negative)
    metrics_registry.add("false_positive", false_positive)
    metrics_registry.add("true_value", gt_positive.shape[0] + gt_negative.shape[0] - false_negative - false_positive)
    metrics_registry.add("total_samples", gt_positive.shape[0] + gt_negative.shape[0])
