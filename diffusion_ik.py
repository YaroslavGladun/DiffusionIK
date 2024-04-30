import torch

from torch.utils.data import Dataset
from common import JointValuesScalerInverse, TransformationUtility
from fk import FK
from typing import Tuple

# https://github.com/bot66/MNISTDiffusion/blob/main/model.py
class RandomDiffusionIKDataset(Dataset):
    def __init__(
            self,
            device,
            batch_size,
            batch_count,
            diffusion_steps=1000,
            beta_small=1e-4,
            beta_large=0.02):
        self.device = device
        self.batch_size = batch_size
        self.batch_count = batch_count
        self.diffusion_steps = diffusion_steps
        self.beta_small = beta_small
        self.beta_large = beta_large

        self.scaler = JointValuesScalerInverse(device)
        self.fk = FK(device)

    def beta(self, time):
        return self.beta_small + (self.beta_large - self.beta_small) * time / self.diffusion_steps

    def alpha(self, time):
        return 1.0 - self.beta(time)

    def __len__(self):
        return self.batch_count * self.batch_size

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        joints = torch.rand(self.batch_size, 7, device=self.device)
        joints = self.scaler(joints)

        R, xyz = self.fk(joints)
        xyz = xyz.squeeze(-1)
        rpy = TransformationUtility.rotation_matrix_to_rpy(R)

        data = torch.cat([joints, xyz, rpy], dim=-1)

        steps = torch.randint(1, self.max_steps, (self.batch_size,))

        # TODO: ...

        X_current = None
        X_last = None

        return X_last, X_current
