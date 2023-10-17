import torch

from src import device


def compute_scalars(array_shape):
    # compute scaling factors based on the dimensions of the array
    return torch.tensor(array_shape, device=device) - 1


def filter_points(points, array_shape):
    # compute the scaling factors
    scalars = compute_scalars(array_shape)

    # scale and round the points
    scaled_points = torch.round(points * scalars).to(torch.int64)

    # create masks to filter out points that are out of array bounds
    dim = len(array_shape)
    masks = [(scaled_points[..., i] >= 0) & (scaled_points[..., i] < array_shape[i]) for i in range(dim)]
    final_mask = torch.stack(masks, dim=-1).all(dim=-1)

    # set out-of-bounds points to zero
    scaled_points[~final_mask] = 0

    return scaled_points, final_mask


def extract_values(array_nd, scaled_points, final_mask):
    # extract the values from 'array_nd' at the coordinates specified in 'scaled_points'
    # set the values to zero where 'final_mask' is False
    values = array_nd[tuple(scaled_points[..., i] for i in range(scaled_points.shape[-1]))]
    values[~final_mask] = 0

    return values.view(scaled_points.shape[:-1] + (1,))


def indicator(points, array_nd):
    # apply the indicator function to get ground truth values
    # filter and scale the points based on the shape of 'array_nd'
    scaled_points, final_mask = filter_points(points, array_nd.shape)

    # extract the values at the scaled and filtered coordinates
    values_at_coords = extract_values(array_nd, scaled_points, final_mask)

    # check if any of the values are 1.0 and reshape the result
    return torch.any(values_at_coords == 1.0, dim=1, keepdim=True).view(points.shape[0], -1).float()
