import torch

from src import device


def get_boxes(batch_size, n_dim):
    # generate random minimum coordinates for boxes
    box_min = torch.rand(batch_size, n_dim, device=device)

    # generate random maximum coordinates for boxes
    box_max = (1 - box_min) * torch.rand(batch_size, n_dim, device=device) + box_min

    # generate boxes by concatenating box_min and box_max
    boxes = torch.cat((box_min, box_max), dim=1)

    return boxes


def sample_boxes(boxes, n_samples):
    batch_size, full_dim = boxes.shape
    n_dim = full_dim // 2
    box_min = boxes[:, :n_dim]
    box_max = boxes[:, n_dim:]

    # compute the range for each dimension for each box
    box_range = box_max - box_min

    # generate random points within each box
    random_points = box_min[:, None, :] + box_range[:, None, :] * torch.rand(batch_size, n_samples, n_dim,
                                                                             device=device)

    return random_points
