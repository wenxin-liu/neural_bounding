import torch

from src import device


# calculate points along rays based on origin, t, and direction
def to_ray(origin, t, direction):
    return origin + t * direction


# generate random directions for rays in n-dimensional space
def generate_random_directions(n_rays, n_dim):
    # generate random angles
    angles = torch.rand(n_rays, n_dim - 1, device=device) * torch.tensor([torch.pi] * (n_dim - 2) + [2 * torch.pi],
                                                                         device=device)
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)

    # generalise for n-dimensions
    directions = torch.ones(n_rays, n_dim, device=device)
    sin_prod = torch.ones(n_rays, device=device)

    for i in range(n_dim - 1):
        directions[:, i] = cos_vals[:, i] * sin_prod
        sin_prod *= sin_vals[:, i]
    directions[:, -1] = sin_prod

    return directions


# sample points along rays in n-dimensional space
def sample_ray(rays, n_samples):
    n_rays = rays.shape[0]
    n_dim = rays.shape[1] // 2

    t = torch.rand([n_rays, n_samples, 1], device=device)

    # compute sample points along each ray
    return to_ray(t=t, origin=rays[:, None, :n_dim], direction=rays[:, None, n_dim:])


# generate random rays
def get_rays(batch_size, n_dim):
    # generate random origins and directions
    random_origins = torch.rand([batch_size, n_dim], device=device)
    random_directions = generate_random_directions(batch_size, n_dim)

    return torch.cat([random_origins, random_directions], dim=1)
