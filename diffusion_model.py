import torch.nn as nn
import torch


class JointValuesScaler(nn.Module):
    def __init__(self, device):
        super(JointValuesScaler, self).__init__()
        self.joints_min = torch.tensor([-6.28, -2.059, -6.28, -0.19, -6.28, -1.69, -6.28]).to(device)
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

    def __init__(self, data_size, condition_size):
        super(DiffusionModel, self).__init__()

        self.fc1 = nn.Linear(data_size + condition_size, 256)
        # self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 256)
        # self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, 256)
        # self.bn3 = nn.BatchNorm1d(256)

        self.fc4 = nn.Linear(256, 256)
        # self.bn4 = nn.BatchNorm1d(256)

        self.f5 = nn.Linear(256, data_size)

    def forward(self, x, condition):
        x = torch.cat((x, condition), 1)
        # x1 = self.bn1(torch.relu(self.fc1(x)))
        # x2 = self.bn2(torch.relu(self.fc2(x1)))
        # x3 = self.bn3(torch.relu(self.fc3(x2 + x1)))
        # x4 = self.bn4(torch.relu(self.fc4(x3 + x2)))
        x1 = torch.relu(self.fc1(x))
        x2 = torch.relu(self.fc2(x1))
        x3 = torch.relu(self.fc3(x2 + x1))
        x4 = torch.relu(self.fc4(x3 + x2))

        x5 = self.f5(x3 + x4)

        return x5
