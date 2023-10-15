import torch

from src.train import train

if __name__ == '__main__':
    torch.manual_seed(0)

    train(object_name="bunny", query="ray", dimension=2)
