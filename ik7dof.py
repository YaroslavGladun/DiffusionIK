import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from torch.utils.data import Dataset
from tqdm import tqdm

from common import JointValuesScalerInverse
from fk import FK
from affine_loss import AffineLoss


class RandomIK6DOFDataset(Dataset):
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
        x = torch.cat([R, t], dim=-1)
        return x, joints


class IK6DOFLoss(nn.Module):

    def __init__(self, device, affine_loss_k=1.0, joints_loss_k=1.0, kld_loss_k=1.0):
        super(IK6DOFLoss, self).__init__()
        self.affine_loss_k = affine_loss_k
        self.joints_loss_k = joints_loss_k
        self.kld_loss_k = kld_loss_k
        self.affine_loss = AffineLoss(alpha=1.0, beta=1.0)
        self.fk = FK(device)

    def forward(self,
                model_input: torch.Tensor,
                joints: torch.Tensor,
                model_joints: torch.Tensor,
                mean: torch.Tensor,
                logvar: torch.Tensor) -> torch.Tensor:
        R_pred, t_pred = self.fk(model_joints)
        R_true, t_true = model_input[:, :9].view(-1, 3, 3), model_input[:, 9:].view(-1, 3, 1)
        affine_loss = self.affine_loss((R_pred, t_pred), (R_true, t_true))
        joints_loss = torch.mean(torch.sqrt(torch.sum(torch.square(joints - model_joints), dim=-1)))
        KLD = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return self.affine_loss_k * affine_loss + self.joints_loss_k * joints_loss + self.kld_loss_k * KLD


class JointsEncoder(nn.Module):

    def __init__(self, device, d_encoder=16, d_model=512):
        assert d_model % 2 == 0

        super(JointsEncoder, self).__init__()
        self.device = device

        d_model_half = int(d_model // 2)

        self.fc1 = nn.Linear(14, d_model_half)
        self.bn1 = nn.BatchNorm1d(d_model_half)

        self.fc2 = nn.Linear(d_model_half, d_model_half)
        self.bn2 = nn.BatchNorm1d(d_model_half)

        self.fc3 = nn.Linear(d_model, d_model_half)
        self.bn3 = nn.BatchNorm1d(d_model_half)

        self.fc4 = nn.Linear(d_model, d_model_half)
        self.bn4 = nn.BatchNorm1d(d_model_half)

        self.fc_mean = nn.Linear(d_model_half, d_encoder)
        self.fc_logvar = nn.Linear(d_model_half, d_encoder)

        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_cos = torch.cos(x)
        x_sin = torch.sin(x)
        x = torch.cat((x_cos, x_sin), dim=1)
        x1 = self.bn1(self.activation(self.fc1(x)))
        x2 = self.bn2(self.activation(self.fc2(x1)))
        x3 = self.bn3(self.activation(self.fc3(torch.cat((x1, x2), dim=1))))
        x4 = self.bn4(self.activation(self.fc4(torch.cat((x2, x3), dim=1))))
        x_mean = self.fc_mean(x4)
        x_logvar = self.fc_logvar(x4)
        return x_mean, x_logvar


class Model(nn.Module):
    def __init__(self, device, d_model=512, d_encoder=16):
        assert d_model % 2 == 0

        super(Model, self).__init__()
        self.device = device

        self.encoder = JointsEncoder(device, d_model=d_model, d_encoder=d_encoder)

        d_model_half = int(d_model // 2)

        self.fc1 = nn.Linear(12 + d_encoder, d_model_half)
        self.bn1 = nn.BatchNorm1d(d_model_half)

        self.fc2 = nn.Linear(d_model_half + d_encoder, d_model_half)
        self.bn2 = nn.BatchNorm1d(d_model_half)

        self.fc3 = nn.Linear(d_model + d_encoder, d_model_half)
        self.bn3 = nn.BatchNorm1d(d_model_half)

        self.fc4 = nn.Linear(d_model + d_encoder, d_model_half)
        self.bn4 = nn.BatchNorm1d(d_model_half)

        self.fc5 = nn.Linear(d_model + d_encoder, d_model_half)
        self.bn5 = nn.BatchNorm1d(d_model_half)

        self.fc6 = nn.Linear(d_model + d_encoder, d_model_half)
        self.bn6 = nn.BatchNorm1d(d_model_half)

        self.fc7 = nn.Linear(d_model + d_encoder, d_model_half)
        self.bn7 = nn.BatchNorm1d(d_model_half)

        self.fc8 = nn.Linear(d_model + d_encoder, d_model_half)
        self.bn8 = nn.BatchNorm1d(d_model_half)

        self.fc_sin = nn.Linear(d_model_half, 7)
        self.fc_cos = nn.Linear(d_model_half, 7)

        self.activation = nn.SiLU()

    def forward(self, x, joints) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: 13 elements tensor:
                    1. joint 0 in range [-pi, pi]
                    2-10. 3x3 rotation matrix
                    11-13. translation vector
        :return: 7 joints of xArm in range [-pi, pi]
        """
        mean, logvar = self.encoder(joints)
        e = self.reparameterization(mean, torch.exp(0.5 * logvar))

        R, t = x[:, :9], x[:, 9:]
        x = torch.cat((R, t), dim=1)
        x1 = self.bn1(self.activation(self.fc1(torch.cat((x, e), dim=1))))
        x2 = self.bn2(self.activation(self.fc2(torch.cat((x1, e), dim=1))))
        x3 = self.bn3(self.activation(self.fc3(torch.cat((x1, x2, e), dim=1))))
        x4 = self.bn4(self.activation(self.fc4(torch.cat((x2, x3, e), dim=1))))
        x5 = self.bn5(self.activation(self.fc5(torch.cat((x3, x4, e), dim=1))))
        x6 = self.bn6(self.activation(self.fc6(torch.cat((x4, x5, e), dim=1))))
        x7 = self.bn7(self.activation(self.fc7(torch.cat((x5, x6, e), dim=1))))
        x8 = self.bn8(self.activation(self.fc8(torch.cat((x6, x7, e), dim=1))))
        x_sin = self.fc_sin(x8)
        x_cos = self.fc_cos(x8)
        x = torch.atan2(x_sin, x_cos)

        return x, mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var * epsilon
        return z


# Setup device and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(device, d_model=2048, d_encoder=16).to(device)

loss_fn = IK6DOFLoss(device)
affine_loss_fn = IK6DOFLoss(device, affine_loss_k=1.0, joints_loss_k=0, kld_loss_k=0)
joints_loss_fn = IK6DOFLoss(device, affine_loss_k=0, joints_loss_k=1.0, kld_loss_k=0)
kld_loss_fn = IK6DOFLoss(device, affine_loss_k=0, joints_loss_k=0, kld_loss_k=1.0)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

lr_space = np.linspace(1e-3, 1e-5, 50)

dataset = RandomIK6DOFDataset(device, 4096, 2048)
test_dataset = RandomIK6DOFDataset(device, 4096, 1)

for epoch in range(100):
    train_loss_accum = 0

    lr = lr_space[epoch]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print(f"Learning rate: {lr:.6f}")
        break

    for i in tqdm(range(dataset.batch_count)):
        x, joints = dataset[i]

        optimizer.zero_grad()
        y_pred, mean, logvar = model(x, joints)
        loss = loss_fn(x, joints, y_pred, mean, logvar)
        loss.backward()
        optimizer.step()

        train_loss_accum += loss.item()

    avg_train_loss = train_loss_accum / dataset.batch_count

    print(f"Epoch {epoch} - Average training loss: {avg_train_loss:.4f}")

    model.eval()
    with torch.no_grad():
        x, joints = test_dataset[0]
        y_pred, mean, logvar = model(x, joints)
        joints_loss = joints_loss_fn(x, joints, y_pred)
        affine_loss = affine_loss_fn(x, joints, y_pred)
        kld_loss = kld_loss_fn(x, joints, y_pred, mean, logvar)
        print(f"Joints loss: {joints_loss.item()}")
        print(f"Affine loss: {affine_loss.item()}")
        print(f"KLD loss: {kld_loss.item()}")
    model.train()
