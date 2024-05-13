import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from torch.utils.data import Dataset
from tqdm import tqdm

from common import JointValuesScalerInverse, TransformationUtility
from fk import FK
from affine_loss import AffineLoss


class IKGTModel(nn.Module):
    def __init__(self, device, d_model=128, one_hot_size=1000):
        assert d_model % 2 == 0

        super(IKGTModel, self).__init__()
        self.device = device
        self.one_hot_size = one_hot_size

        d_model_half = int(d_model // 2)

        self.fc1 = nn.Linear(12, d_model_half)
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

        self.fc_sin = nn.Linear(d_model_half, 7)
        self.fc_cos = nn.Linear(d_model_half, 7)

        self.activation = nn.SiLU()

    def forward(self, x) -> torch.Tensor:
        """
        :param x: shape (batch_size, 12), R and t
        :return: 7 joints of xArm
        """
        x1 = self.bn1(self.activation(self.fc1(x)))
        x2 = self.bn2(self.activation(self.fc2(x1)))
        x3 = self.bn3(self.activation(self.fc3(torch.cat((x1, x2), dim=1))))
        x4 = self.bn4(self.activation(self.fc4(torch.cat((x2, x3), dim=1))))
        x5 = self.bn5(self.activation(self.fc5(torch.cat((x3, x4), dim=1))))
        x6 = self.bn6(self.activation(self.fc6(torch.cat((x4, x5), dim=1))))
        x7 = self.bn7(self.activation(self.fc7(torch.cat((x5, x6), dim=1))))
        x8 = self.bn8(self.activation(self.fc8(torch.cat((x6, x7), dim=1))))
        x_sin = self.fc_sin(x8)
        x_cos = self.fc_cos(x8)
        x = torch.atan2(x_sin, x_cos)

        return x
