import torch
import torch.nn as nn
from typing import Tuple
from torch.utils.data import Dataset
from tqdm import tqdm

from common import JointValuesScalerInverse, TransformationUtility
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
        y = joints[:, 1:]
        x = torch.cat([joints[:, 0].view(-1, 1), R, t], dim=-1)

        return x, y


class IK6DOFLoss(nn.Module):

    def __init__(self, device):
        super(IK6DOFLoss, self).__init__()
        self.affine_loss = AffineLoss(alpha=1.0, beta=1.0)
        self.fk = FK(device)

    def forward(self, model_input: torch.Tensor, model_output: torch.Tensor) -> torch.Tensor:
        R_pred, t_pred = self.fk(model_output)
        R_true, t_true = model_input[:, 1:10].view(-1, 3, 3), model_input[:, 10:].view(-1, 3, 1)
        return self.affine_loss((R_pred, t_pred), (R_true, t_true))


class Model(nn.Module):
    def __init__(self, device, d_model=128):
        assert d_model % 2 == 0

        super(Model, self).__init__()
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

        self.fc5 = nn.Linear(d_model, d_model_half)
        self.bn5 = nn.BatchNorm1d(d_model_half)

        self.fc6 = nn.Linear(d_model, d_model_half)
        self.bn6 = nn.BatchNorm1d(d_model_half)

        self.fc7 = nn.Linear(d_model, d_model_half)
        self.bn7 = nn.BatchNorm1d(d_model_half)

        self.fc8 = nn.Linear(d_model, d_model_half)
        self.bn8 = nn.BatchNorm1d(d_model_half)

        self.fc_sin = nn.Linear(d_model_half, 6)
        self.fc_cos = nn.Linear(d_model_half, 6)

        self.activation = nn.SiLU()

    def forward(self, x) -> torch.Tensor:
        """

        :param x:
        :return: 7 joints of xArm
        """
        j0, R, t = x[:, 0], x[:, 1:10], x[:, 10:]
        j0 = j0.view(-1, 1)
        j_cos = torch.cos(j0)
        j_sin = torch.sin(j0)
        x = torch.cat((j_cos, j_sin, R, t), dim=1)
        x1 = self.bn1(self.activation(self.fc1(x)))
        x2 = self.bn2(self.activation(self.fc2(x1)))
        x3 = self.bn3(self.activation(self.fc3(torch.cat((x1, x2), dim=1))))
        x4 = self.bn4(self.activation(self.fc4(torch.cat((x2, x3), dim=1))))
        x5 = self.bn5(self.activation(self.fc5(torch.cat((x3, x4), dim=1))))
        x6 = self.bn6(self.activation(self.fc6(torch.cat((x4, x5), dim=1))))
        x7 = self.bn7(self.activation(self.fc7(torch.cat((x5, x6), dim=1))))
        x8 = self.bn8(self.activation(self.fc8(torch.cat((x6, x7), dim=1))))
        x_sin = self.fc_sin(x8)
        x_cos = self.fc_cos(x8)
        x = torch.atan2(x_sin, x_cos)

        x = torch.cat((j0, x), dim=1)

        return x


# Setup device and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(device).to(device)

# Loss function and optimizer
loss_fn = IK6DOFLoss(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

dataset = RandomIK6DOFDataset(device, 10000, 2000)

for epoch in range(100):
    train_loss_accum = 0

    for i in tqdm(range(dataset.batch_count)):
        x, y = dataset[i]

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(x, y_pred)
        loss.backward()
        optimizer.step()

        train_loss_accum += loss.item()

    avg_train_loss = train_loss_accum / dataset.batch_count

    print(f"Epoch {epoch} - Average training loss: {avg_train_loss:.4f}")
