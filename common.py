import torch
import torch.nn as nn

from math import pi
from typing import Tuple

JOINTS_LOWER_LIMIT = [-2.0 * pi, -2.059, -2.0 * pi, -0.19198, -2.0 * pi, -1.69297, -2.0 * pi]
JOINTS_UPPER_LIMIT = [2.0 * pi, 2.0944, 2.0 * pi, 3.927, 2.0 * pi, pi, 2.0 * pi]


class JointValuesClamp(nn.Module):
    def __init__(self, device):
        super(JointValuesClamp, self).__init__()
        self.joints_min = torch.tensor(JOINTS_LOWER_LIMIT).to(device)
        self.joints_max = torch.tensor(JOINTS_UPPER_LIMIT).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Clamps the input tensor to the joint limits
        :param x: shape (N, 7) tensor of joint values
        :return: shape (N, 7) tensor of joint values clamped to the joint limits
        """
        return torch.max(torch.min(x, self.joints_max), self.joints_min)


class JointValuesScaler(nn.Module):
    def __init__(self, device):
        super(JointValuesScaler, self).__init__()
        self.joints_min = torch.tensor(JOINTS_LOWER_LIMIT).to(device)
        self.joints_max = torch.tensor(JOINTS_UPPER_LIMIT).to(device)

    def forward(self, x):
        return (x - self.joints_min) / (self.joints_max - self.joints_min)


class JointValuesScalerInverse(nn.Module):

    def __init__(self, device):
        super(JointValuesScalerInverse, self).__init__()
        self.joints_min = torch.tensor(JOINTS_LOWER_LIMIT).to(device)
        self.joints_max = torch.tensor(JOINTS_UPPER_LIMIT).to(device)

    def forward(self, x):
        return x * (self.joints_max - self.joints_min) + self.joints_min


class TransformationUtility:
    @staticmethod
    def xyz_rpy_to_torch_affine(xyzrpy: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts a tensor of xyzrpy values to a tensor of 4x4 affine matrices
        :param xyzrpy: shape (N, 6) tensor of xyzrpy values
        :return: two torch tensors;
        The first tensor is the rotation matrix and the second tensor is the translation vector
        """
        x, y, z, roll, pitch, yaw = xyzrpy[:, 0], xyzrpy[:, 1], xyzrpy[:, 2], xyzrpy[:, 3], xyzrpy[:, 4], xyzrpy[:, 5]

        Rz = TransformationUtility.rotation_z(yaw)
        Ry = TransformationUtility.rotation_y(pitch)
        Rx = TransformationUtility.rotation_x(roll)

        R = torch.matmul(torch.matmul(Rz, Ry), Rx)
        t = torch.stack([x, y, z], dim=1).unsqueeze(-1)

        return R, t

    @staticmethod
    def rotation_x(angle: torch.Tensor) -> torch.Tensor:
        """
        Returns the rotation matrix around the x-axis
        :param angle: the angle in radians; shape (...,)
        :return: the rotation matrix; shape (..., 3, 3)
        """
        cx = torch.cos(angle)
        sx = torch.sin(angle)

        return torch.stack([
            torch.stack([torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)], dim=-1),
            torch.stack([torch.zeros_like(cx), cx, -sx], dim=-1),
            torch.stack([torch.zeros_like(cx), sx, cx], dim=-1)
        ], dim=-2)

    @staticmethod
    def rotation_y(angle: torch.Tensor) -> torch.Tensor:
        """
        Returns the rotation matrix around the y-axis
        :param angle: the angle in radians; shape (...,)
        :return: the rotation matrix; shape (..., 3, 3)
        """
        cy = torch.cos(angle)
        sy = torch.sin(angle)

        return torch.stack([
            torch.stack([cy, torch.zeros_like(cy), sy], dim=-1),
            torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)], dim=-1),
            torch.stack([-sy, torch.zeros_like(cy), cy], dim=-1)
        ], dim=-2)

    @staticmethod
    def rotation_z(angle: torch.Tensor) -> torch.Tensor:
        """
        Returns the rotation matrix around the z-axis
        :param angle: the angle in radians; shape (...,)
        :return: the rotation matrix; shape (..., 3, 3)
        """
        cz = torch.cos(angle)
        sz = torch.sin(angle)

        zero = torch.zeros_like(cz)
        one = torch.ones_like(cz)

        return torch.stack([
            torch.stack([cz, -sz, zero], dim=-1),
            torch.stack([sz, cz, zero], dim=-1),
            torch.stack([zero, zero, one], dim=-1)
        ], dim=-2)

    @staticmethod
    def rotation_to_affine(rotation: torch.Tensor) -> torch.Tensor:
        """
        Converts a tensor of rotation matrices to a tensor of 4x4 affine matrices
        :param rotation: shape (N, 3, 3) tensor of rotation matrices
        :return: shape (N, 4, 4) tensor of affine matrices
        """
        affine_matrices = torch.zeros(rotation.shape[0], 4, 4, device=rotation.device, dtype=rotation.dtype)
        affine_matrices[:, :3, :3] = rotation
        affine_matrices[:, 3, 3] = 1
        return affine_matrices

    @staticmethod
    def rotation_matrix_to_rpy(rotation: torch.Tensor) -> torch.Tensor:
        """
        Converts a tensor of rotation matrices to a tensor of rpy values
        :param rotation: shape (N, 3, 3) tensor of rotation matrices
        :return: shape (N, 3) tensor of rpy values
        """
        rpy = torch.zeros(rotation.shape[0], 3, device=rotation.device, dtype=rotation.dtype)
        rpy[:, 0] = torch.atan2(rotation[:, 2, 1], rotation[:, 2, 2])
        rpy[:, 1] = -torch.asin(rotation[:, 2, 0])
        rpy[:, 2] = torch.atan2(rotation[:, 1, 0], rotation[:, 0, 0])
        return rpy