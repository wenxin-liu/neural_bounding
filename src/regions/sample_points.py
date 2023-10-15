import torch


def generate_points(n_points, n_dim):
    random_points = torch.rand([n_points, n_dim])

    return random_points
