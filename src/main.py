import torch

from src.model.train_ours_neural_2d import train_ours_neural_2d_point

if __name__ == '__main__':
    torch.manual_seed(0)

    train_ours_neural_2d_point()
