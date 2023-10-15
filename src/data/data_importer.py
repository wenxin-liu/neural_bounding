import torch
import matplotlib.pyplot as plt
import numpy as np

from src import device
from src.data import binvox_rw


# function to load 2D data
def load_data_2d(object_name):
    return torch.tensor(plt.imread(f'{object_name}_32.png')[:, :, 3] >= 0.5).float().to(device)


def load_binvox(fn):
    with open(fn, 'rb') as fin:
        out = binvox_rw.read_as_3d_array(fin)
        return np.array(out.data)


# function to load 3D data
def load_data_3d(object_name):
    numpy_array = np.where(load_binvox(f'{object_name}.binvox') == True, 1.0, 0.0)
    return torch.tensor(numpy_array, dtype=torch.float32).to(device)


# function to load 4D data
def load_data_4d(object_name):
    numpy_array = np.load(f'{object_name}_4d.npy')
    return torch.tensor(numpy_array, dtype=torch.float32).to(device)
