import torch


# AABox - axis-aligned bounding box implementation
def calculate_AABox(gt_positive, gt_negative, metrics_registry):
    # find aabb min and max coordinates
    aabb_min = torch.min(gt_positive, dim=0)[0]
    aabb_max = torch.max(gt_positive, dim=0)[0]

    # take all the true negative points and calculate which fall inside the aabb
    is_inside_aabb = torch.all(aabb_min <= gt_negative, dim=1) & torch.all(gt_negative <= aabb_max, dim=1)

    # this produces the false positive values for aabb
    fp_aabb = torch.sum(is_inside_aabb).item()

    # take all the true positive points and calculate which fall outside the aabb
    is_outside_aabb = torch.any(aabb_min > gt_positive, dim=1) | torch.any(gt_positive > aabb_max, dim=1)

    # this produces the false negative values for aabb - should be 0
    fn_aabb = torch.sum(is_outside_aabb).item()

    # save AABox baseline results
    metrics_registry.metrics_registry["AABox"] = {
        "class weight": "N/A",
        "iteration": "N/A",
        "false negatives": fn_aabb,
        "false positives": fp_aabb,
        "true values": gt_positive.shape[0] + gt_negative.shape[0] - fp_aabb - fn_aabb,
        "total samples": gt_positive.shape[0] + gt_negative.shape[0],
        "loss": "N/A"
    }
