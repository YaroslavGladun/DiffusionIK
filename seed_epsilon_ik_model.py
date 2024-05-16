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
        super(SeedEpsilonIKModel, self).__init__()

        # pose, joint_cos, joint_sin, epsilon
        self.fc1 = nn.Linear(12 + 7 + 7 + 1, d_model)
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

        self.fc7 = nn.Linear(d_model, d_model)
        self.bn7 = nn.BatchNorm1d(d_model)

        self.fc8 = nn.Linear(d_model, d_model)
        self.bn8 = nn.BatchNorm1d(d_model)

        self.fc_sin = nn.Linear(d_model, 7)
        self.fc_cos = nn.Linear(d_model, 7)

        self.activation = nn.SiLU()

    def forward(self, pose, seed, cosine_diff) -> torch.Tensor:
        """
        :param pose: shape (batch_size, 12)
        :param seed: shape (batch_size, 7)
        :param cosine_diff: shape (batch_size, 1)
        :return: 7 joints of xArm
        """
        x0 = torch.cat((pose, torch.cos(seed), torch.sin(seed), cosine_diff), dim=1)
        x1 = self.bn1(self.activation(self.fc1(x0)))
        x2 = self.bn2(self.activation(self.fc2(x1)))
        x3 = self.bn3(self.activation(self.fc3(x1 + x2)))
        x4 = self.bn4(self.activation(self.fc4(x2 + x3)))
        x5 = self.bn5(self.activation(self.fc5(x3 + x4)))
        x6 = self.bn6(self.activation(self.fc6(x4 + x5)))
        x7 = self.bn7(self.activation(self.fc7(x5 + x6)))
        x8 = self.bn8(self.activation(self.fc8(x6 + x7)))
        # x = self.fc_out(torch.cat((x7, x8), dim=1))
        x_cos = self.fc_cos(x7 + x8)
        x_sin = self.fc_sin(x7 + x8)
        x = torch.atan2(x_sin, x_cos)
        # return seed + x
        return x

# Epoch 14 - Average testing loss: 0.1260
