import torch.nn as nn
import torch


class CVAEEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CVAEEncoder, self).__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc4 = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(256)

        self.mean = nn.Linear(256, hidden_size)
        self.logvar = nn.Linear(256, hidden_size)

    def forward(self, x):
        x1 = self.bn1(torch.relu(self.fc1(x)))
        x2 = self.bn2(torch.relu(self.fc2(x1)))
        x3 = self.bn3(torch.relu(self.fc3(x2 + x1)))
        x4 = self.bn4(torch.relu(self.fc4(x3 + x2)))

        mean = self.mean(x3 + x4)
        logvar = self.logvar(x3 + x4)

        return mean, logvar


class CVAEDecoder(nn.Module):

    def __init__(self, output_size, hidden_size):
        super(CVAEDecoder, self).__init__()

        self.fc1 = nn.Linear(hidden_size, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc4 = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(256)

        self.fc5 = nn.Linear(256, output_size)

    def forward(self, x):
        x1 = self.bn1(torch.relu(self.fc1(x)))
        x2 = self.bn2(torch.relu(self.fc2(x1)))
        x3 = self.bn3(torch.relu(self.fc3(x2 + x1)))
        x4 = self.bn4(torch.relu(self.fc4(x3 + x2)))
        x5 = self.fc5(x4 + x3)
        return x5


class CVAE(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, condition_size):
        super(CVAE, self).__init__()
        self.encoder = CVAEEncoder(input_size, hidden_size)
        self.decoder = CVAEDecoder(output_size, hidden_size + condition_size)
        self.condition_size = condition_size

    def forward(self, x, condition):
        mean, logvar = self.encoder(x)
        x = self.reparameterization(mean, torch.exp(0.5 * logvar))
        x = torch.cat((x, condition), dim=1)
        x = self.decoder(x)
        return x, mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z
