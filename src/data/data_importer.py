import torch
import matplotlib.pyplot as plt

from src import device


def load_data_2d(object_name):
    return torch.tensor(plt.imread(f'{object_name}_32.png')[:, :, 3] >= 0.5).float().to(device)
