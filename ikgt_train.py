import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from torch.utils.data import Dataset
from tqdm import tqdm
from fk import FK

from common import TransformationUtility
from ikgt_model import IKGTModel
from ikgt_dataset import IKGTDataset
from affine_loss import AffineLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
model = IKGTModel(device).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

lr_space = np.linspace(1e-3, 2 * 1e-5, 200)

dataset = IKGTDataset(device, 2048, 128)
test_dataset = IKGTDataset(device, 2048, 1)

fk = FK(device)
affine_loss = AffineLoss(alpha=1.0, beta=1.0)

for epoch in range(1000):
    train_loss_accum = 0

    lr = 1e-3
    # lr = lr_space[epoch - 1]
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = lr
    #     print(f"Learning rate: {lr:.6f}")
    #     break

    for i in tqdm(range(dataset.batch_count)):
        x, y = dataset[i]

        optimizer.zero_grad()
        loss = model(x, y)
        loss.backward()
        optimizer.step()

        train_loss_accum += loss.item()

    avg_train_loss = train_loss_accum / dataset.batch_count

    print(f"Epoch {epoch} - Average training loss: {avg_train_loss:.4f}")

    x, y = test_dataset[0]
    y_hat = model.generate(x)
    R_hat, t_hat = fk(y_hat)
    R, t = x[:, :9].view(-1, 3, 3), x[:, 9:].view(-1, 3)
    affine_loss_value = affine_loss((R, t), (R_hat, t_hat))
    print(f"Affine loss: {affine_loss_value:.4f}")
