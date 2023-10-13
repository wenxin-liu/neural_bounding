import torch.nn as nn
import torch


class OursNeural2D(nn.Module):
    def __init__(self, input_dim):
        super(OursNeural2D, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(self.input_dim, 25)
        self.fc2 = nn.Linear(25, 25)
        self.fc3 = nn.Linear(25, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
