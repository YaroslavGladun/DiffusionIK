import torch
import torch.nn as nn
from typing import Tuple
from torch.utils.data import Dataset
from common import JointValuesScalerInverse, TransformationUtility


class FK(nn.Module):
    def __init__(self, device):
        super(FK, self).__init__()

        transformations = torch.tensor([
            [0, 0, 0.267, 0, 0, 0],
            [0, 0, 0, -1.5708, 0, 0],
            [0, -0.293, 0, 1.5708, 0, 0],
            [0.0525, 0, 0, 1.5708, 0, 0],
            [0.0775, -0.3425, 0, 1.5708, 0, 0],
            [0, 0, 0, 1.5708, 0, 0],
            [0.076, 0.097, 0, -1.5708, 0, 0]
        ]).to(device)

        self.joint_transforms = TransformationUtility.xyz_rpy_to_torch_affine(transformations)

    def forward(self, x: torch.Tensor, begin: int = 0, end: int = 7) -> Tuple[torch.Tensor, torch.Tensor]:
        R_result = torch.eye(3, device=x.device).unsqueeze(0).expand(x.size(0), 3, 3)
        t_result = torch.zeros(x.size(0), 3, 1, device=x.device)

        joint_rotations = TransformationUtility.rotation_z(x)

        for i in range(begin, end):
            R, t = self.joint_transforms[0][i], self.joint_transforms[1][i]
            t_result = torch.matmul(R_result, t) + t_result
            R_result = torch.matmul(R_result, R)
            R_j = joint_rotations[:, i]
            R_result = torch.matmul(R_result, R_j)

        return R_result, t_result