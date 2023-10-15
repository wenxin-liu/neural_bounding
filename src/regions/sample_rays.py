import torch

from src import device


# calculate points along rays based on origin, t, and direction.
def _to_ray(origin, t, direction):
    return origin + t * direction


# generate random directions for rays in n-dimensional space
def _generate_random_directions(n_rays, n_dim):
    # generate random angles
    angles = torch.rand(n_rays, n_dim - 1, device=device) * torch.tensor([torch.pi] * (n_dim - 2) + [2 * torch.pi],
                                                                         device=device)
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)

    if n_dim == 2:
        directions = torch.stack([cos_vals[:, 0], sin_vals[:, 0]], dim=-1)
    elif n_dim == 3:
        x1 = cos_vals[:, 0]
        x2 = sin_vals[:, 0] * cos_vals[:, 1]
        x3 = sin_vals[:, 0] * sin_vals[:, 1]
        directions = torch.stack([x1, x2, x3], dim=-1).to(device)
    elif n_dim == 4:
        x1 = cos_vals[:, 0]
        x2 = sin_vals[:, 0] * cos_vals[:, 1]
        x3 = sin_vals[:, 0] * sin_vals[:, 1] * cos_vals[:, 2]
        x4 = sin_vals[:, 0] * sin_vals[:, 1] * sin_vals[:, 2]
        directions = torch.stack([x1, x2, x3, x4], dim=-1).to(device)
    else:
        raise ValueError("only 2D, 3D, and 4D are currently supported")

    return directions


# sample points along rays in n-dimensional space.
def sample_ray(rays, n_samples_on_ray):
    n_rays = rays.shape[0]
    n_dim = rays.shape[1] // 2

    t = torch.rand([n_rays, n_samples_on_ray, 1], device=device)

    # compute sample points along each ray
    rays_expanded = rays[:, None].expand(-1, n_samples_on_ray, -1)

    return _to_ray(t=t, origin=rays_expanded[..., :n_dim], direction=rays_expanded[..., n_dim:])


def generate_rays(n_dim, n_rays):
    # generate random origins and directions
    random_origins = torch.rand([n_rays, n_dim], device=device)
    random_directions = _generate_random_directions(n_rays, n_dim)

    random_rays = torch.cat([random_origins, random_directions], dim=1)

    return random_rays
