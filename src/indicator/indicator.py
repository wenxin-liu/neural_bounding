import torch


# compute scaling factors based on array dimensions
def compute_scalars(array_shape):
    return torch.tensor(array_shape) - 1


# scale and filter points based on array dimensions
def filter_points(points, array_shape):
    # compute scaling factors
    scalars = compute_scalars(array_shape)

    # scale and round the points
    scaled_points = torch.round(points * scalars)

    # clamp the points to be within the array dimensions
    min_tensor = torch.zeros_like(scalars)
    max_tensor = scalars
    clamped_points = torch.min(torch.max(scaled_points, min_tensor), max_tensor)

    return torch.round(clamped_points).to(int)


# retrieve values at filtered coordinates
def extract_values(array_nd, points):
    values = array_nd[tuple(points[..., i] for i in range(points.shape[-1]))]
    return values.view(points.shape[:-1] + (1,))


# indicator function to get ground truth values
def indicator(points, array_nd):
    # scale and filter the points
    scaled_points = filter_points(points, array_nd.shape)

    # extract values at the scaled and filtered coordinates
    values_at_coords = extract_values(array_nd, scaled_points)

    return torch.any(values_at_coords == 1.0, dim=1, keepdim=True).view(points.shape[0], -1).float()
