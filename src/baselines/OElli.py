import torch
import torch.optim as optim


# OElli - aka. oriented ellipsoid, non-axis-aligned ellipsoid or NonAAEllipsoid - implementation
def generate_ellipsoid_params(gt_positive_transformed):
    # use the mean of the transformed points as the centre
    centre_coords = torch.mean(gt_positive_transformed, dim=0)

    # use the standard deviation of the transformed points as the axis lengths
    # scaled down to allow room for the optimisation process to adjust the axis lengths
    axis_lengths = torch.std(gt_positive_transformed, dim=0) * 0.8

    # concatenate to form the ellipsoid parameters
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

    is_inside_or_on = distance_from_surface <= 1e-3

    num_inside_or_on = torch.sum(is_inside_or_on).item()
    num_outside = len(is_inside_or_on) - num_inside_or_on

    return num_inside_or_on, num_outside


def calculate_OElli(gt_negative, gt_positive, metrics_registry, dim):
    # clean metrics registry
    metrics_registry.reset_metrics()

    covariance = torch.cov(gt_positive.T)

    eigen_values, eigen_vectors = torch.linalg.eig(covariance)
    eigen_values = eigen_values.real
    eigen_vectors = eigen_vectors.real

    gt_positive_transformed = gt_positive @ eigen_vectors
    gt_negative_transformed = gt_negative @ eigen_vectors

    ellipsoid_params = generate_ellipsoid_params(gt_positive_transformed)
    optimizer = optim.Adam([ellipsoid_params], lr=0.001)

    # Optimization loop
    for i in range(30000):
        optimizer.zero_grad()
        loss = loss_fn(gt_positive_transformed, ellipsoid_params, dim)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Iteration {i}, Loss {loss.item()}")

        if loss == 0.:
            print(f"Iteration {i}, Loss {loss.item()}")
            break

    false_positives, true_negative = is_inside_ellipsoid(gt_negative_transformed, ellipsoid_params, dim)
    true_positive, false_negatives = is_inside_ellipsoid(gt_positive_transformed, ellipsoid_params, dim)

    metrics_registry.register_counter_metric("false_negative")
    metrics_registry.register_counter_metric("false_positive")
    metrics_registry.register_counter_metric("true_value")
    metrics_registry.register_counter_metric("total_samples")

    metrics_registry.add("false_negative", false_negatives)
    metrics_registry.add("false_positive", false_positives)
    metrics_registry.add("true_value", true_negative + true_positive)
    metrics_registry.add("total_samples", false_negatives + false_positives + true_negative + true_positive)