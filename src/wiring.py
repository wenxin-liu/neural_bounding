from pathlib import Path

from src import device
from src.data.data_importer import load_data_2d, load_data_3d, load_data_4d
from src.indicator.indicator import indicator
from src.ours_neural.nn_model_2d import OursNeural2D
from src.ours_neural.nn_model_3d import OursNeural3D
from src.ours_neural.nn_model_4d import OursNeural4D, OursNeural4DPlane
from src.regions.sample_boxes import get_boxes, sample_boxes
from src.regions.sample_planes import get_planes, sample_planes
from src.regions.sample_points import get_points
from src.regions.sample_rays import get_rays, sample_ray


def get_source_data(object_name, dimension):
    dataset_path = Path(__file__).resolve().parents[1] / 'data' / f'{dimension}d'
    if dimension == 2:
        return load_data_2d(f'{dataset_path}/{object_name}.png')
    elif dimension == 3:
        return load_data_3d(f'{dataset_path}/{object_name}.binvox')
    elif dimension == 4:
        return load_data_4d(f'{dataset_path}/{object_name}.npy')
    else:
        raise ValueError(f"Unsupported dimension: {dimension}")


def get_training_data(data, query, dimension, n_regions, n_samples=1):
    if query == 'point':
        # generate n_regions number of random points
        features = get_points(n_regions, dimension).to(device)

        # generate the corresponding targets using indicator function
        targets = indicator(features, data).to(device)
    elif query == 'ray':
        # generate n_regions number of random rays
        features = get_rays(n_dim=dimension, batch_size=n_regions).to(device)

        # generate n_samples number of random points along the ray
        sampled_rays = sample_ray(rays=features, n_samples=n_samples).to(device)

        # generate the corresponding targets using indicator function
        targets = indicator(sampled_rays, data).to(device)
    elif query == 'plane':
        # generate n_regions number of random hyperplanes
        features = get_planes(n_dim=dimension, batch_size=n_regions).to(device)

        # generate n_samples number of random points on the hyperplane surface
        sampled_planes = sample_planes(planes=features, n_samples=n_samples).to(device)

        # generate the corresponding targets using indicator function
        targets = indicator(sampled_planes, data).to(device)
    elif query == 'box':
        # generate n_regions number of random hypercubes
        features = get_boxes(n_dim=dimension, batch_size=n_regions).to(device)

        # generate n_samples number of random points inside the hypercube
        sampled_boxes = sample_boxes(boxes=features, n_samples=n_samples).to(device)

        # generate the corresponding targets using indicator function
        targets = indicator(sampled_boxes, data).to(device)
    else:
        raise ValueError(f"{query} not a valid query type")

    return features, targets


# select model based on query and dimension
def get_model(query, dimension):
    if dimension == 2 and query == "point":
        return OursNeural2D(dimension).to(device)
    elif dimension == 2 and query != "point":
        return OursNeural2D(dimension * 2).to(device)
    elif dimension == 3 and query == "point":
        return OursNeural3D(dimension).to(device)
    elif dimension == 3 and query != "point":
        return OursNeural3D(dimension * 2).to(device)
    elif dimension == 4 and query == "point":
        return OursNeural4D(dimension).to(device)
    elif dimension == 4 and query == "plane":
        return OursNeural4DPlane(dimension * 2).to(device)
    elif dimension == 4 and (query == "ray" or query == "box"):
        return OursNeural4D(dimension * 2).to(device)
    else:
        raise ValueError(f"Unsupported task: {query} and {dimension}")
