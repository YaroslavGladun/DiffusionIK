import torch
import torch.nn as nn

from tqdm import tqdm
from typing import Tuple, Dict

from fk import FK
from probabilistic_ik_config import ProbabilisticIKConfig
from common import JointValuesScalerInverse


class ProbabilisticIKModel(nn.Module):

    def __init__(self, config: ProbabilisticIKConfig, device: torch.device):
        super(ProbabilisticIKModel, self).__init__()
        self.config = config
        self.device = device

        self.models = nn.ModuleList([ProbabilisticIKJointModel(config) for _ in range(7)])
        self.criterion = nn.CrossEntropyLoss()
        self.scaler_inverse = JointValuesScalerInverse(device)
        self.fk = FK(device)

    def forward(self, x: Dict[str, torch.Tensor]):
        """
        :param x:
        :return: loss
        """
        loss = 0
        for i in range(7):
            y_target = x["joints_one_hot"][:, i, :]
            y_pred = self.models[i](x["pose"][:, i, :])
            loss += self.criterion(y_pred, y_target)
        return loss

    def get_ik(self, x: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        :param x:
        :return: joints, joints_one_hot
        """
        target_pose_global = x["pose"][:, 0, :]
        target_pose_global_R = target_pose_global[:, :9].view(-1, 3, 3)
        target_pose_global_t = target_pose_global[:, 9:].view(-1, 3, 1)

        i_link_pose_R = torch.eye(3, device=self.device).unsqueeze(0).expand(x["joints_one_hot"].size(0), 3, 3)
        i_link_pose_t = torch.zeros(x["joints_one_hot"].size(0), 3, 1, device=self.device)

        result_scaled = torch.zeros(x["joints_one_hot"].size(0), 7, device=self.device)
        result = torch.zeros(x["joints_one_hot"].size(0), 7, device=self.device)
        for i in range(7):
            # model input is i_index_pose.inv * target_pose_global
            # model_input_R, model_input_t = ...
            i_link_pose_R_inv = i_link_pose_R.inverse()
            i_link_pose_t_inv = -i_link_pose_R_inv @ i_link_pose_t
            model_input_R = i_link_pose_R_inv @ target_pose_global_R
            model_input_t = i_link_pose_R_inv @ target_pose_global_t + i_link_pose_t_inv

            model_inout = torch.cat([model_input_R.view(-1, 9), model_input_t.view(-1, 3)], dim=-1)

            result_scaled[:, i] = self.models[i](model_inout).argmax(dim=-1) / self.config.trace_steps
            result = self.scaler_inverse(result_scaled)
            i_link_pose_R, i_link_pose_t = self.fk(result, end=i + 1)

        return result


class ProbabilisticIKJointModel(nn.Module):
    def __init__(self, config: ProbabilisticIKConfig):
        super(ProbabilisticIKJointModel, self).__init__()

        self.config = config

        self.fc1 = nn.Linear(12, config.d_model)
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

        self.fc_out = nn.Linear(config.d_model, config.trace_steps)

        self.activation = nn.SiLU()
        self.softmax = nn.Softmax()

    def forward(self, pose) -> torch.Tensor:
        """
        :param pose: shape (batch_size, 12)
        :return: 7 joints of xArm
        """

        x1 = self.bn1(self.activation(self.fc1(pose)))
        x2 = self.bn2(self.activation(self.fc2(x1)))
        x3 = self.bn3(self.activation(self.fc3(x1 + x2)))
        x4 = self.bn4(self.activation(self.fc4(x2 + x3)))
        x5 = self.bn5(self.activation(self.fc5(x3 + x4)))
        x6 = self.bn6(self.activation(self.fc6(x4 + x5)))
        x7 = self.bn7(self.activation(self.fc7(x5 + x6)))
        x8 = self.bn8(self.activation(self.fc8(x6 + x7)))
        x9 = self.bn9(self.activation(self.fc9(x7 + x8)))
        x10 = self.bn10(self.activation(self.fc10(x8 + x9)))

        x_out = self.softmax(self.fc_out(x10))

        return x_out
