import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from torch.utils.data import Dataset
from tqdm import tqdm

from common import JointValuesScalerInverse, TransformationUtility
from fk import FK
from affine_loss import AffineLoss


class SeedEpsilonIKEncoder(nn.Module):

    def __init__(self, device, d_model=128, n_layers=6):
        super(SeedEpsilonIKEncoder, self).__init__()

        self.fk = FK(device)

        self.d_model = d_model
        self.n_layers = n_layers

        # pose, joints, joints_cos, joints_sin, seed_pose
        self.fc1 = nn.Linear(12 + 7 + 7 + 7 + 12, d_model)
        self.bn1 = nn.BatchNorm1d(d_model)

        self.fc2 = nn.Linear(d_model, d_model)
        self.bn2 = nn.BatchNorm1d(d_model)

        self.fc3 = nn.Linear(d_model, d_model)
        self.bn3 = nn.BatchNorm1d(d_model)

        self.fc4 = nn.Linear(d_model, d_model)
        self.bn4 = nn.BatchNorm1d(d_model)

        self.fc5 = nn.Linear(d_model, d_model)
        self.bn5 = nn.BatchNorm1d(d_model)

        self.fc6 = nn.Linear(d_model, d_model)
        self.bn6 = nn.BatchNorm1d(d_model)

        self.activation = nn.SiLU()

    def forward(self, pose, seed) -> torch.Tensor:
        """
        :param pose: shape (batch_size, 12)
        :param seed: shape (batch_size, 7)
        :return: 7 joints of xArm
        """
        seed_cos = torch.cos(seed)
        seed_sin = torch.sin(seed)
        seed_R, seed_t = self.fk(seed)
        seed_pose = torch.cat([seed_R.view(-1, 9), seed_t.view(-1, 3)], dim=-1)
        x = torch.cat((pose, seed, seed_cos, seed_sin, seed_pose), dim=1)
        x1 = self.bn1(self.activation(self.fc1(x)))
        x2 = self.bn2(self.activation(self.fc2(x1)))
        x3 = self.bn3(self.activation(self.fc3(x1 + x2)))
        x4 = self.bn4(self.activation(self.fc4(x2 + x3)))
        x5 = self.bn5(self.activation(self.fc5(x3 + x4)))
        x6 = self.bn6(self.activation(self.fc6(x4 + x5)))

        return x6


class SeedEpsilonIKModel(nn.Module):
    def __init__(self, device, d_model=128, n_encoder_layers=6):
        super(SeedEpsilonIKModel, self).__init__()

        self.encoder = SeedEpsilonIKEncoder(device, d_model=d_model, n_layers=n_encoder_layers)

        # pose, joint_cos, joint_sin, epsilon
        self.fc1 = nn.Linear(d_model + 1, d_model)
        self.bn1 = nn.BatchNorm1d(d_model)

        self.fc2 = nn.Linear(d_model + 1, d_model)
        self.bn2 = nn.BatchNorm1d(d_model)

        self.fc3 = nn.Linear(d_model + 1, d_model)
        self.bn3 = nn.BatchNorm1d(d_model)

        self.fc4 = nn.Linear(d_model + 1, d_model)
        self.bn4 = nn.BatchNorm1d(d_model)

        self.fc5 = nn.Linear(d_model + 1, d_model)
        self.bn5 = nn.BatchNorm1d(d_model)

        self.fc6 = nn.Linear(d_model + 1, d_model)
        self.bn6 = nn.BatchNorm1d(d_model)

        self.fc_out = nn.Linear(d_model + 1, 7)

        self.activation = nn.SiLU()

    def forward(self, pose, seed, angle_diff) -> torch.Tensor:
        """
        :param pose: shape (batch_size, 12)
        :param seed: shape (batch_size, 7)
        :param angle_diff: shape (batch_size, 1)
        :return: 7 joints of xArm
        """

        encoder_output = self.encoder(pose, seed)
        x1 = self.bn1(self.activation(self.fc1(torch.cat((encoder_output, angle_diff), dim=1))))
        x2 = self.bn2(self.activation(self.fc2(torch.cat((x1, angle_diff), dim=1))))
        x3 = self.bn3(self.activation(self.fc3(torch.cat((x1 + x2, angle_diff), dim=1))))
        x4 = self.bn4(self.activation(self.fc4(torch.cat((x2 + x3, angle_diff), dim=1))))
        x5 = self.bn5(self.activation(self.fc5(torch.cat((x3 + x4, angle_diff), dim=1))))
        x6 = self.bn6(self.activation(self.fc6(torch.cat((x4 + x5, angle_diff), dim=1))))

        x = self.fc_out(torch.cat((x5 + x6, angle_diff), dim=1))
        x = x / torch.norm(x, dim=-1, keepdim=True)

        return seed + angle_diff * x
