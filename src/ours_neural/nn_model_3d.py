import torch.nn as nn
import torch


class OursNeural3D(nn.Module):
    def __init__(self, input_dim):
        super(OursNeural3D, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(self.input_dim, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
