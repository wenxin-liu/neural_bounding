def extract_ground_truth_classes(features, targets):
    # remove singleton dimensions from targets
    targets = targets.squeeze(dim=-1)

    # extract ground truth positive and negative features based on the targets
    gt_positive_mask = targets == 1
    gt_negative_mask = targets == 0

    # filter features based on the masks
    gt_positive = features[gt_positive_mask]
    gt_negative = features[gt_negative_mask]

    return {
        "gt_positive": gt_positive,
        "gt_negative": gt_negative
    }
