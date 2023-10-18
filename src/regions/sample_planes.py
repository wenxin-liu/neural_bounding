import torch
from torch import Tensor

from src import device


# generate random normals for planes in n-dimensional space
def generate_random_normals(n_planes, n_dim):
    # generate random angles
    angles = torch.rand(n_planes, n_dim - 1, device=device) * torch.tensor([torch.pi] * (n_dim - 2) + [2 * torch.pi],
                                                                           device=device)
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)

    # generalise for n-dimensions
    normals = torch.ones(n_planes, n_dim, device=device)
    sin_prod = torch.ones(n_planes, device=device)

    for i in range(n_dim - 1):
        normals[:, i] = cos_vals[:, i] * sin_prod
        sin_prod *= sin_vals[:, i]
    normals[:, -1] = sin_prod

    return normals


# generate random normals and origin points for hyperplanes in n-dimensional space
def get_planes(batch_size: int, n_dim: int) -> Tensor:
    # generate random normals and normalise them
    normals = generate_random_normals(n_planes=batch_size, n_dim=n_dim)

    # generate random origin points
    points = torch.rand(batch_size, n_dim, device=device)

    # concatenate normals and points
    return torch.cat((normals, points), dim=1)


# find n_samples random points on each plane defined by its normal vector and a point
def find_points_on_plane(normals: Tensor, points: Tensor, n_samples: int) -> Tensor:
    # generate random points in the same dimension as the plane
    random_points = torch.rand(normals.shape[0], n_samples, normals.shape[1], device=device) * 6 - 3

    # calculate the constant term for the plane equation
    constant_terms = torch.sum(normals * points, dim=1, keepdim=True)

    # add a small epsilon to avoid division by zero
    epsilon = 1e-10

    # solve for the first coordinate (x) of the random point on the plane
    x = (constant_terms - torch.sum(normals[:, None, 1:] * random_points[:, :, 1:], dim=2)) / (
            normals[:, None, 0] + epsilon)

    # concatenate x with the other coordinates to get the full point
    return torch.cat((x.unsqueeze(2), random_points[:, :, 1:]), dim=2)


# sample random points on a batch of planes
def sample_planes(planes: Tensor, n_samples: int) -> Tensor:
    # extract normals and points from the planes
    dim = planes.shape[1] // 2
    normals = planes[..., :dim]
    points = planes[..., dim:]

    # find multiple points on each plane
    return find_points_on_plane(normals, points, n_samples)
