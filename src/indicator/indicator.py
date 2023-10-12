import torch

from src import device


def compute_scalars(array_shape, dim):
    # compute scaling factors based on array dimensions
    if dim == 4:
        return torch.tensor([array_shape[0] - 1] + [array_shape[1] - 1] * (dim - 1), dtype=torch.float32)
    return torch.tensor([array_shape[0] - 1] * dim, dtype=torch.float32)


def filter_points(points, array_shape):
    # scale and filter points based on array dimensions
    dim = len(array_shape)
    scalars = compute_scalars(array_shape, dim).to(device)
    scaled_points = torch.round(points * scalars).to(torch.int64)
    masks = [(scaled_points[..., i] >= 0) & (scaled_points[..., i] < array_shape[i]) for i in range(dim)]
    final_mask = torch.stack(masks, dim=-1).all(dim=-1)
    scaled_points[~final_mask] = 0
    return scaled_points, final_mask


def extract_values(array_nd, scaled_points, final_mask):
    # retrieve values at filtered coordinates
    values = array_nd[tuple(scaled_points[..., i] for i in range(scaled_points.shape[-1]))]
    values[~final_mask] = 0
    return values.view(scaled_points.shape[:-1] + (1,))


def indicator(points, array_nd):
    # indicator function to get ground truth values
    array_shape = array_nd.shape
    scaled_points, final_mask = filter_points(points, array_shape)
    values_at_coords = extract_values(array_nd, scaled_points, final_mask)

    if values_at_coords.shape[0] == 1:
        return torch.any(values_at_coords == 1.0, dim=-1, keepdim=True).float()

    return torch.any(values_at_coords == 1.0, dim=1, keepdim=True).float()
