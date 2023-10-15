import torch.nn as nn
import torch


class OursNeural4D(nn.Module):
    def __init__(self, input_dim):
        super(OursNeural4D, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(self.input_dim, 75)
        self.fc2 = nn.Linear(75, 75)
        self.fc3 = nn.Linear(75, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class OursNeural4DPlane(nn.Module):
    def __init__(self, input_dim):
        super(OursNeural4DPlane, self).__init__()
        self.input_dim = input_dim
        self.fc1 = nn.Linear(self.input_dim, 75)
        self.fc2 = nn.Linear(75, 75)
        self.fc3 = nn.Linear(75, 75)
        self.fc4 = nn.Linear(75, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x

