import torch
from typing import Tuple
from torch.utils.data import Dataset

from common import JointValuesScalerInverse
from fk import FK


class SeedEpsilonIKDataset(Dataset):
    def __init__(self, device, batch_size, batch_count):
        self.device = device
        self.batch_size = batch_size
        self.batch_count = batch_count
        self.scaler = JointValuesScalerInverse(device)
        self.fk = FK(device)

    def __len__(self):
        return self.batch_count * self.batch_size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Joints scaling using atan2, so angles are in range [-pi, pi]
        :param index: index of the batch
        :return:
        """
        target_joints = torch.rand(self.batch_size, 7, device=self.device)
        target_joints = self.scaler(target_joints)
        target_joints = torch.atan2(torch.sin(target_joints), torch.cos(target_joints))
        target_pose_R, target_pose_t = self.fk(target_joints)
        target_pose = torch.cat([target_pose_R.view(-1, 9), target_pose_t.view(-1, 3)], dim=-1)
        epsilon_lower = 0.01
        epsilon_upper = torch.pi / 4
        epsilon = torch.rand(self.batch_size, 1, device=self.device) * (epsilon_upper - epsilon_lower) + epsilon_lower
        noise = (2 * torch.rand(self.batch_size, 7, device=self.device) - 1) * epsilon
        seed_joints = target_joints + noise
        seed_joints = torch.max(torch.min(seed_joints, torch.tensor([torch.pi] * 7, device=self.device)),
                                torch.tensor([-torch.pi] * 7, device=self.device))
        return target_pose, seed_joints, epsilon
