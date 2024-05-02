import torch
import torch.nn as nn

from torch.utils.data import Dataset
from common import JointValuesScalerInverse, TransformationUtility
from fk import FK
from typing import Tuple


# https://github.com/bot66/MNISTDiffusion/blob/main/model.py
# https://medium.com/@mickael.boillaud/denoising-diffusion-model-from-scratch-using-pytorch-658805d293b4


class DiffusionModel:

    def __init__(self, device, diffusion_steps=1000, beta_small=1e-4, beta_large=0.01):
        self.device = device
        self.diffusion_steps = diffusion_steps
        self.beta_small = beta_small
        self.beta_large = beta_large

        # Cosine annealing for beta values
        timesteps = torch.arange(diffusion_steps, device=device)
        self.betas = self.beta_small + (self.beta_large - self.beta_small) * \
                     (1 - torch.cos(timesteps / diffusion_steps * torch.pi)) / 2

        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, 0).unsqueeze(-1)

        print(f"Min alpha: {self.alphas_cumprod.min()}")
        print(f"Max alpha: {self.alphas_cumprod.max()}")

    def beta(self, steps: torch.Tensor) -> torch.Tensor:
        """
        :param steps: shape (N,)
        :return: shape (N,)
        """
        return self.beta_small + (self.beta_large - self.beta_small) * steps / self.diffusion_steps

    def alpha(self, steps):
        return 1.0 - self.beta(steps)


class RandomDiffusionIKDataset(Dataset):
    def __init__(
            self,
            device,
            batch_size,
            batch_count,
            diffusion_model: DiffusionModel):
        self.device = device
        self.batch_size = batch_size
        self.batch_count = batch_count
        self.diffusion_model = diffusion_model

        self.scaler = JointValuesScalerInverse(device)
        self.fk = FK(device)

    def __len__(self):
        return self.batch_count * self.batch_size

    def make_batch_x0(self):
        joints = torch.rand(self.batch_size, 7, device=self.device)
        joints = self.scaler(joints)

        R, xyz = self.fk(joints)
        xyz = xyz.squeeze(-1)
        # rpy = TransformationUtility.rotation_matrix_to_rpy(R)
        # reshape R
        # R = R.view(-1, 9)

        data = torch.cat([joints, xyz], dim=-1)

        return data

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        x0 = self.make_batch_x0()
        t = torch.randint(1, self.diffusion_model.diffusion_steps, (self.batch_size,)).to(self.device)
        epsilon = torch.randn_like(x0)
        alpha_cumprod = self.diffusion_model.alphas_cumprod[t - 1]
        x_t = torch.sqrt(alpha_cumprod) * x0 + torch.sqrt(1 - alpha_cumprod) * epsilon
        x_t_with_steps = torch.cat([x_t, t.unsqueeze(-1).float() / self.diffusion_model.diffusion_steps], dim=-1)
        return x_t_with_steps, epsilon


class Model(nn.Module):
    def __init__(self, device, diffusion_model: DiffusionModel, d_model=128):
        assert d_model % 2 == 0

        super(Model, self).__init__()
        self.device = device
        self.diffusion_model = diffusion_model

        d_model_half = int(d_model // 2)

        self.fc1 = nn.Linear(11, d_model_half)
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

        self.fc_out = nn.Linear(d_model_half, 10)

        self.activation = nn.SiLU()

    def forward(self, x):
        x1 = self.bn1(self.activation(self.fc1(x)))
        x2 = self.bn2(self.activation(self.fc2(x1)))
        x3 = self.bn3(self.activation(self.fc3(torch.cat((x1, x2), dim=1))))
        x4 = self.bn4(self.activation(self.fc4(torch.cat((x2, x3), dim=1))))
        x5 = self.bn5(self.activation(self.fc5(torch.cat((x3, x4), dim=1))))
        x6 = self.bn6(self.activation(self.fc6(torch.cat((x4, x5), dim=1))))
        x7 = self.bn7(self.activation(self.fc7(torch.cat((x5, x6), dim=1))))
        x8 = self.bn8(self.activation(self.fc8(torch.cat((x6, x7), dim=1))))
        x = self.fc_out(x8)

        return x

    def sample(self, batch_size):
        x_t = torch.randn(batch_size, 19, device=self.device)
        for t in range(self.diffusion_model.diffusion_steps, 0, -1):
            if t > 1:
                z = torch.randn(batch_size, 19, device=self.device)
            else:
                z = torch.zeros(batch_size, 19, device=self.device)
            x_t_minus_1 = () / torch.sqrt()
