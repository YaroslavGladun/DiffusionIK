import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from torch.utils.data import Dataset
from tqdm import tqdm

from common import JointValuesScalerInverse, TransformationUtility
from fk import FK
from affine_loss import AffineLoss


class SeedEpsilonIKModel(nn.Module):
    def __init__(self, d_model=128):
        assert d_model % 2 == 0

        super(SeedEpsilonIKModel, self).__init__()
        d_model_half = int(d_model // 2)

        # pose, joint_cos, joint_sin, epsilon
        self.fc1 = nn.Linear(12 + 7 + 7 + 1, d_model_half)
        self.bn1 = nn.BatchNorm1d(d_model_half)

        self.fc2 = nn.Linear(d_model_half, d_model_half)
        self.bn2 = nn.BatchNorm1d(d_model_half)

        self.fc3 = nn.Linear(d_model, d_model_half)
        self.bn3 = nn.BatchNorm1d(d_model_half)

        self.fc4 = nn.Linear(d_model, d_model_half)
        self.bn4 = nn.BatchNorm1d(d_model_half)

        self.fc5 = nn.Linear(d_model, d_model_half)
        self.bn5 = nn.BatchNorm1d(d_model_half)

        self.fc6 = nn.Linear(d_model, d_model_half)
        self.bn6 = nn.BatchNorm1d(d_model_half)

        self.fc7 = nn.Linear(d_model, d_model_half)
        self.bn7 = nn.BatchNorm1d(d_model_half)

        self.fc8 = nn.Linear(d_model, d_model_half)
        self.bn8 = nn.BatchNorm1d(d_model_half)

        self.fc_out = nn.Linear(d_model_half, 7)

        self.activation = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, pose, seed, epsilon) -> torch.Tensor:
        """
        :param pose: shape (batch_size, 12)
        :param seed: shape (batch_size, 7)
        :param epsilon: shape (batch_size, 1)
        :return: 7 joints of xArm
        """
        x = torch.cat((pose, torch.cos(seed), torch.sin(seed), epsilon), dim=1)
        x1 = self.bn1(self.activation(self.fc1(x)))
        x2 = self.bn2(self.activation(self.fc2(x1)))
        x3 = self.bn3(self.activation(self.fc3(torch.cat((x1, x2), dim=1))))
        x4 = self.bn4(self.activation(self.fc4(torch.cat((x2, x3), dim=1))))
        x5 = self.bn5(self.activation(self.fc5(torch.cat((x3, x4), dim=1))))
        x6 = self.bn6(self.activation(self.fc6(torch.cat((x4, x5), dim=1))))
        x7 = self.bn7(self.activation(self.fc7(torch.cat((x5, x6), dim=1))))
        x8 = self.bn8(self.activation(self.fc8(torch.cat((x6, x7), dim=1))))
        x = seed + epsilon * (2 * self.sigmoid(self.fc_out(x8)) - 1)
        return x
