import numpy as np
from PIL import Image
import torch


def load_data_2d(filename):
    # read image, convert to grayscale, and transform to a numpy array
    image_array = np.array(Image.open(filename).convert('L'))

    # convert to binary and return as a PyTorch tensor
    return torch.tensor(image_array >= 127.5, dtype=torch.float32)
