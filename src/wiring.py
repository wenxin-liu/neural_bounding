from src import device
from src.data.data_importer import load_data_2d, load_data_3d, load_data_4d
from src.indicator.indicator import indicator
from src.ours_neural.nn_model_2d import OursNeural2D
from src.ours_neural.nn_model_3d import OursNeural3D
from src.ours_neural.nn_model_4d import OursNeural4D, OursNeural4DPlane
from src.regions.sample_planes import get_planes, sample_planes
from src.regions.sample_points import get_points
from src.regions.sample_rays import get_rays, sample_ray


def get_source_data(object_name, dimension):
    if dimension == 2:
        return load_data_2d(object_name)
    elif dimension == 3:
        return load_data_3d(object_name)
    elif dimension == 4:
        return load_data_4d(object_name)
    else:
        raise ValueError(f"Unsupported dimension: {dimension}")


def get_training_data(data, query, dimension, n_objects, n_samples=1):
    if query == 'point':
        features = get_points(n_objects, dimension).to(device)

        # generate the corresponding targets using indicator function
        targets = indicator(features, data).to(device)
    elif query == 'ray':
        # generate n_sample of random sample points
        features = get_rays(n_dim=dimension, batch_size=n_objects).to(device)

        sampled_rays = sample_ray(rays=features, n_samples=n_samples).to(device)

        # generate the corresponding targets using indicator function
        targets = indicator(sampled_rays, data).to(device)
    elif query == 'plane':
        # generate n_sample of random sample points
        features = get_planes(n_dim=dimension, batch_size=n_objects).to(device)

        sampled_planes = sample_planes(planes=features, n_samples=n_samples).to(device)

        # generate the corresponding targets using indicator function
        targets = indicator(sampled_planes, data).to(device)
    else:
        raise ValueError("other queries not yet supported")

    return features, targets


def get_model(query, dimension):
    if dimension == 2 and query == "point":
        return OursNeural2D(dimension).to(device)
    elif dimension == 2 and query != "point":
        return OursNeural2D(dimension*2).to(device)
    elif dimension == 3 and query == "point":
        return OursNeural3D(dimension).to(device)
    elif dimension == 3 and query != "point":
        return OursNeural3D(dimension*2).to(device)
    elif dimension == 4 and query == "point":
        return OursNeural4D(dimension).to(device)
    elif dimension == 4 and query == "plane":
        return OursNeural4DPlane(dimension*2).to(device)
    elif dimension == 4 and (query == "ray" or query == "box"):
        return OursNeural4D(dimension*2).to(device)
    else:
        raise ValueError(f"Unsupported task: {query} and {dimension}")

