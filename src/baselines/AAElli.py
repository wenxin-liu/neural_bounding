import torch
import torch.optim as optim

from src import device


def generate_ellipsoid_params(dimensions, centre=0.5, radius=0.3):
    # create the centre coordinates
    centre_coords = torch.full((dimensions,), centre, dtype=torch.float32, device=device)

    # create the axis lengths (radius)
    axis_lengths = torch.full((dimensions,), radius, dtype=torch.float32, device=device)

    # concatenate to form the ellipsoid parameters
    # initialize ellipsoid parameters: [x_center, y_center, z_center, a, b, c]
    ellipsoid_params = torch.cat((centre_coords, axis_lengths))

    # set requires_grad to True for gradient-based optimization
    ellipsoid_params.requires_grad = True

    return ellipsoid_params


def distance_to_ellipsoid(points, ellipsoid_params, dim):
    center, axis_lengths = ellipsoid_params[:dim], ellipsoid_params[dim:]
    normed_diff = (points - center) / axis_lengths
    return torch.norm(normed_diff, dim=1) - 1


def loss_fn(points, ellipsoid_params, dim):
    distances = distance_to_ellipsoid(points, ellipsoid_params, dim)
    return torch.sum(torch.relu(distances))


def is_inside_ellipsoid(points, params, dim):
    center, axis_lengths = params[:dim], params[dim:]
    normed_diff = (points - center) / axis_lengths
    distance_from_surface = torch.norm(normed_diff, dim=1) - 1

    is_inside_or_on = distance_from_surface <= 1e-4

    num_inside_or_on = torch.sum(is_inside_or_on).item()
    num_outside = len(is_inside_or_on) - num_inside_or_on

    return num_inside_or_on, num_outside


# AAElli - axis-aligned ellipsoid implementation
def calculate_AAElli(gt_negative, gt_positive, metrics_registry, dim):
    ellipsoid_params = generate_ellipsoid_params(dimensions=dim)
    optimizer = optim.Adam([ellipsoid_params], lr=0.01)

    # Optimization loop
    for i in range(30000):
        optimizer.zero_grad()
        loss = loss_fn(gt_positive, ellipsoid_params, dim)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Iteration {i}, Loss {loss.item()}")

        if loss == 0.:
            print(f"Iteration {i}, Loss {loss.item()}")
            break

    false_positives, true_negative = is_inside_ellipsoid(gt_negative, ellipsoid_params, dim)
    true_positive, false_negatives = is_inside_ellipsoid(gt_positive, ellipsoid_params, dim)

    # save AAElli baseline results
    metrics_registry.metrics_registry["AAElli"] = {
        "class weight": "N/A",
        "iteration": "N/A",
        "false negatives": false_negatives,
        "false positives": false_positives,
        "true values": true_negative + true_positive,
        "total samples": false_negatives + false_positives + true_negative + true_positive,
        "loss": "N/A"
    }
