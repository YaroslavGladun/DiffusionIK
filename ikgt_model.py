import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from torch.utils.data import Dataset
from tqdm import tqdm

from common import JointValuesScalerInverse, TransformationUtility
from fk import FK
from affine_loss import AffineLoss


class IndexMapper(nn.Module):

    def __init__(self, min_value: float = -np.pi, max_value: float = np.pi, size: int = 1000):
        super(IndexMapper, self).__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.min_value) / (self.max_value - self.min_value)
        x = torch.clamp(x, 0, 1)
        x = x * (self.size - 1)
        x = x.long()
        return x


class IndexMapperInverse(nn.Module):

    def __init__(self, min_value: float = -np.pi, max_value: float = np.pi, size: int = 1000):
        super(IndexMapperInverse, self).__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = x / (self.size - 1)
        x = x * (self.max_value - self.min_value)
        x = x + self.min_value
        return x


class OneHotEncoder(nn.Module):

    def __init__(self, size: int = 1000):
        super(OneHotEncoder, self).__init__()
        self.size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: shape (..., N)
        :return: shape (..., N, size)
        """
        shape = x.size()
        x = x.view(-1)
        x = torch.nn.functional.one_hot(x, num_classes=self.size)
        x = x.view(*shape, self.size)
        return x


class IKGTBlock(nn.Module):
    def __init__(self, input_size, d_model=128):
        assert d_model % 2 == 0

        super(IKGTBlock, self).__init__()
        d_model_half = int(d_model // 2)

        self.fc1 = nn.Linear(input_size, d_model_half)
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

        self.fc_out = nn.Linear(d_model_half, 1000)

        self.activation = nn.SiLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x) -> torch.Tensor:
        """
        :param x:
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
        x_out = self.fc_out(x8)
        x_out = self.softmax(x_out)

        return x_out


class IKGTModel(nn.Module):
    def __init__(self, device):
        super(IKGTModel, self).__init__()
        self.device = device
        self.index_mapper = IndexMapper()
        self.index_mapper_inverse = IndexMapperInverse()
        self.one_hot_encoder = OneHotEncoder()

        self.block0 = IKGTBlock(12 + 2 * 0)
        self.block1 = IKGTBlock(12 + 2 * 1)
        self.block2 = IKGTBlock(12 + 2 * 2)
        self.block3 = IKGTBlock(12 + 2 * 3)
        self.block4 = IKGTBlock(12 + 2 * 4)
        self.block5 = IKGTBlock(12 + 2 * 5)
        self.block6 = IKGTBlock(12 + 2 * 6)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y) -> torch.Tensor:
        x0, y0 = self.make_pair_for_model(x, y, 0)
        y0_hat = self.block0(x0)
        loss0 = self.criterion(y0_hat, y0.argmax(dim=-1))

        x1, y1 = self.make_pair_for_model(x, y, 1)
        y1_hat = self.block1(x1)
        loss1 = self.criterion(y1_hat, y1.argmax(dim=-1))

        x2, y2 = self.make_pair_for_model(x, y, 2)
        y2_hat = self.block2(x2)
        loss2 = self.criterion(y2_hat, y2.argmax(dim=-1))

        x3, y3 = self.make_pair_for_model(x, y, 3)
        y3_hat = self.block3(x3)
        loss3 = self.criterion(y3_hat, y3.argmax(dim=-1))

        x4, y4 = self.make_pair_for_model(x, y, 4)
        y4_hat = self.block4(x4)
        loss4 = self.criterion(y4_hat, y4.argmax(dim=-1))

        x5, y5 = self.make_pair_for_model(x, y, 5)
        y5_hat = self.block5(x5)
        loss5 = self.criterion(y5_hat, y5.argmax(dim=-1))

        x6, y6 = self.make_pair_for_model(x, y, 6)
        y6_hat = self.block6(x6)
        loss6 = self.criterion(y6_hat, y6.argmax(dim=-1))

        loss = loss0 + loss1 + loss2 + loss3 + loss4 + loss5 + loss6

        return loss

    def make_pair_for_model(self, x, y, model_index):
        """
        :param x: shape (batch_size, 12)
        :param y: shape (batch_size, 7)
        :param model_index: int
        :return: x and y for the model with index model_index
        """
        if model_index == 0:
            return x, self.one_hot_encoder(self.index_mapper(y[:, 0]))
        else:
            x = torch.cat((x, y[..., :model_index]), dim=1)
            return x, self.one_hot_encoder(self.index_mapper(y[:, model_index]))

    def generate(self, x):
        """
        :param x: shape (batch_size, 12)
        :return:
        """
        x0 = x
        y0 = self.index_mapper_inverse(self.block0(x).argmax(dim=-1)).view(-1, 1)

        x1 = torch.cat((x0, y0), dim=1)
        y1 = self.index_mapper_inverse(self.block1(x1).argmax(dim=-1)).view(-1, 1)

        x2 = torch.cat((x1, y1), dim=1)
        y2 = self.index_mapper_inverse(self.block2(x2).argmax(dim=-1)).view(-1, 1)

        x3 = torch.cat((x2, y2), dim=1)
        y3 = self.index_mapper_inverse(self.block3(x3).argmax(dim=-1)).view(-1, 1)

        x4 = torch.cat((x3, y3), dim=1)
        y4 = self.index_mapper_inverse(self.block4(x4).argmax(dim=-1)).view(-1, 1)

        x5 = torch.cat((x4, y4), dim=1)
        y5 = self.index_mapper_inverse(self.block5(x5).argmax(dim=-1)).view(-1, 1)

        x6 = torch.cat((x5, y5), dim=1)
        y6 = self.index_mapper_inverse(self.block6(x6).argmax(dim=-1)).view(-1, 1)

        y = torch.cat((y0, y1, y2, y3, y4, y5, y6), dim=1)

        return y
