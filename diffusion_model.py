import torch.nn as nn
import torch
import math


def d(x):
    return x * math.pi / 180


class JointValuesScaler(nn.Module):
    def __init__(self, device):
        super(JointValuesScaler, self).__init__()
        self.joints_min = torch.tensor([
            d(-360),
            ]).to(device)
        self.joints_max = torch.tensor([6.28, 2.09, 6.28, 3.92, 6.28, 3.14, 6.28]).to(device)

    def forward(self, x):
        return (x - self.joints_min) / (self.joints_max - self.joints_min)


class JointValuesScalerInverse(nn.Module):

    def __init__(self, device):
        super(JointValuesScalerInverse, self).__init__()
        self.joints_min = torch.tensor([-6.28, -2.059, -6.28, -0.19, -6.28, -1.69, -6.28]).to(device) - 0.01
        self.joints_max = torch.tensor([6.28, 2.09, 6.28, 3.92, 6.28, 3.14, 6.28]).to(device) + 0.01

    def forward(self, x):
        return x * (self.joints_max - self.joints_min) + self.joints_min


class DiffusionModel(nn.Module):
    LAYERS = 6
    N_HIDDEN = 256

    def __init__(self, device, data_size, condition_size):
        super(DiffusionModel, self).__init__()

        self.f0 = nn.Linear(data_size + condition_size, self.N_HIDDEN).to(device)

        self.f = []
        for i in range(self.LAYERS):
            self.f.append(nn.Linear(self.N_HIDDEN, self.N_HIDDEN).to(device))

        self.f_out = nn.Linear(self.N_HIDDEN, data_size).to(device)

    def forward(self, x, condition):
        x = torch.cat((x, condition), 1)

        x = torch.relu(self.f0(x))
        for i in range(self.LAYERS):
            x = torch.relu(self.f[i](x))
        x = self.f_out(x)

        return x
