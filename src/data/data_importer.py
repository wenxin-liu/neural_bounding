import os
import shutil
import zipfile

import torch
import numpy as np
import matplotlib.pyplot as plt

from src import device
from src.data import binvox_rw

from PIL import Image


# methods to read image from file, convert to grayscale, then convert to binary
# def load_data_2d(filename):
#     filename = f'{filename}_32_rot90.png'
#     image = Image.open(filename).convert('L')
#
#     # convert grayscale image into a numpy array
#     return torch.from_numpy(np.where(np.array(image) >= 127.5, 1., 0.)).to(device)

# function to load 2D data
def load_data_2d(object_name):
    return torch.tensor(plt.imread(object_name)[:, :, 3] >= 0.5, device=device).float()


def load_binvox(fn):
    with open(fn, 'rb') as fin:
        out = binvox_rw.read_as_3d_array(fin)
        return np.array(out.data)


# function to load 3D data
def load_data_3d(object_name):
    numpy_array = np.where(load_binvox(object_name) == True, 1.0, 0.0)
    return torch.tensor(numpy_array, dtype=torch.float32, device=device)


# function to load 4D data
def load_data_4d(object_name):
    numpy_array = np.load(object_name)
    return torch.tensor(numpy_array, dtype=torch.float32, device=device)


def import_dataset_from_gdrive(resources_path, dim):
    if os.path.isdir(resources_path):
        print("Dataset already downloaded, skipping.")
        return

    else:
        try:
            resources_path.mkdir(parents=True, exist_ok=True)

            import gdown
            if dim == 2:
                # download 2d dataset from google drive
                url = 'https://drive.google.com/uc?id=1chJPVUQ7FUbzj6U8q9FleM9D-n4s0L7d'
                output = '2d.zip'
                zip_path = '2d.zip'
            elif dim == 3:
                # download 3d dataset from google drive
                url = 'https://drive.google.com/uc?id=1hFtd2g8wfO9AwfZdg-EdMNlaNcG7Tl3j'
                output = '3d.zip'
                zip_path = '3d.zip'
            elif dim == 4:
                # download 4d dataset from google drive
                url = 'https://drive.google.com/uc?id=1BnnQwBkG8JlekcJw2RpSg1TCdGqiUJMI'
                output = '4d.zip'
                zip_path = '4d.zip'
            else:
                return

            gdown.download(url, output, quiet=False)

            # zip Extraction
            extract_path = 'temp_folder'
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

            # move files
            destination_folder = resources_path
            for filename in os.listdir(extract_path):
                shutil.move(os.path.join(extract_path, filename), os.path.join(destination_folder, filename))

            # remove temp files
            os.remove(zip_path)
            shutil.rmtree(extract_path)

        except Exception as ex:
            shutil.rmtree(resources_path)
            raise ex

