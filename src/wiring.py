from src import device
from src.data.data_importer import load_data_2d
from src.indicator.indicator import indicator
from src.ours_neural.nn_model_2d import OursNeural2D
from src.regions.sample_points import generate_points
from src.regions.sample_rays import generate_rays, sample_ray


def get_source_data(object_name, dimension):
    if dimension == 2:
        return load_data_2d(object_name)


def get_training_data(query, dimension, data, n_objects, n_samples=1):
    if query == 'point':
        features = generate_points(n_objects, dimension).to(device)

        # generate the corresponding targets using indicator function
        targets = indicator(features, data).to(device)
    elif query == 'ray':
        # generate n_sample of random sample points
        features = generate_rays(n_dim=dimension, n_rays=n_objects).to(device)

        sampled_rays = sample_ray(features, n_samples_on_ray=n_samples).to(device)

        # generate the corresponding targets using indicator function
        targets = indicator(sampled_rays, data).to(device)
    else:
        raise ValueError("other queries not yet supported")

    return features, targets


def get_model(query, dimension):
    if dimension == 2 and query == "point":
        return OursNeural2D(dimension).to(device)
    else:
        return OursNeural2D(dimension*2).to(device)
