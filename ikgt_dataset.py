import torch
from typing import Tuple
from torch.utils.data import Dataset

from common import JointValuesScalerInverse
from fk import FK


class IKGTDataset(Dataset):
    def __init__(self, device, batch_size, batch_count):
        self.device = device
        self.batch_size = batch_size
        self.batch_count = batch_count
        self.scaler = JointValuesScalerInverse(device)
        self.fk = FK(device)

    def __len__(self):
        return self.batch_count * self.batch_size

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Joints scaling using atan2, so angles are in range [-pi, pi]
        :param index: index of the batch
        :return:
        """
        joints = torch.rand(self.batch_size, 7, device=self.device)
        joints = self.scaler(joints)
        joints = torch.atan2(torch.sin(joints), torch.cos(joints))
        R, t = self.fk(joints)
        R = R.view(-1, 9)
        t = t.view(-1, 3)
        return torch.cat([R, t], dim=-1), joints
