import torch
import torch.nn as nn

from typing import Tuple

from fk import FK
from seed_epsilon_ik_config import SeedEpsilonIKConfig


class SeedEpsilonIKModel(nn.Module):
    def __init__(self, device, config: SeedEpsilonIKConfig):
        super(SeedEpsilonIKModel, self).__init__()

        self.config = config

        self.fk = FK(device)

        # self.encoder = SeedEpsilonIKEncoder(device, config)

        # target_pose, seed_joints, seed_joints_cos, seed_joints_sin, seed_pose
        self.fc1 = nn.Linear(12 + 7 + 7 + 7 + 12, config.d_model)
        self.bn1 = nn.BatchNorm1d(config.d_model)

        self.fc2 = nn.Linear(config.d_model, config.d_model)
        self.bn2 = nn.BatchNorm1d(config.d_model)

        self.fc3 = nn.Linear(config.d_model, config.d_model)
        self.bn3 = nn.BatchNorm1d(config.d_model)

        self.fc4 = nn.Linear(config.d_model, config.d_model)
        self.bn4 = nn.BatchNorm1d(config.d_model)

        self.fc5 = nn.Linear(config.d_model, config.d_model)
        self.bn5 = nn.BatchNorm1d(config.d_model)

        self.fc6 = nn.Linear(config.d_model, config.d_model)
        self.bn6 = nn.BatchNorm1d(config.d_model)

        self.fc7 = nn.Linear(config.d_model, config.d_model)
        self.bn7 = nn.BatchNorm1d(config.d_model)

        self.fc8 = nn.Linear(config.d_model, config.d_model)
        self.bn8 = nn.BatchNorm1d(config.d_model)

        self.fc9 = nn.Linear(config.d_model, config.d_model)
        self.bn9 = nn.BatchNorm1d(config.d_model)

        self.fc10 = nn.Linear(config.d_model, config.d_model)
        self.bn10 = nn.BatchNorm1d(config.d_model)

        self.fc11 = nn.Linear(config.d_model, config.d_model)
        self.bn11 = nn.BatchNorm1d(config.d_model)

        self.fc12 = nn.Linear(config.d_model, config.d_model)
        self.bn12 = nn.BatchNorm1d(config.d_model)

        self.fc_joints = nn.Linear(config.d_model, 7)

        self.activation = nn.SiLU()
        self.tanh = nn.Tanh()

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
        x7 = self.bn7(self.activation(self.fc7(x5 + x6)))
        x8 = self.bn8(self.activation(self.fc8(x6 + x7)))
        x9 = self.bn9(self.activation(self.fc9(x7 + x8)))
        x10 = self.bn10(self.activation(self.fc10(x8 + x9)))
        x11 = self.bn11(self.activation(self.fc11(x9 + x10)))
        x12 = self.bn12(self.activation(self.fc12(x10 + x11 + x6)))

        x = self.tanh(self.fc_joints(x11 + x12))
        x = x / torch.norm(x, dim=-1, keepdim=True)

        return seed + self.config.max_seed_dist * x

    def find_nearest_solution(self, pose, seed) -> torch.Tensor:
        seed_R, seed_t = self.fk(seed)
        target_R, target_t = pose[:, :9].view(-1, 3, 3), pose[:, 9:].view(-1, 3)
