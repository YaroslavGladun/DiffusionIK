import numpy as np
import torch
import torch.nn as nn
from typing import Tuple
from torch.utils.data import Dataset
from tqdm import tqdm

from common import TransformationUtility
from ikgt_model import IKGTModel
from ikgt_dataset import IKGTDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
model = IKGTModel(device, 2048).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

lr_space = np.linspace(1e-3, 2 * 1e-5, 200)

dataset = IKGTDataset(device, 2048, 128)

for epoch in range(100):
    train_loss_accum = 0

    lr = lr_space[epoch - 1]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print(f"Learning rate: {lr:.6f}")
        break

    for i in tqdm(range(dataset.batch_count)):
        x, y = dataset[i]

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y, y_pred)
        loss.backward()
        optimizer.step()

        train_loss_accum += loss.item()

    avg_train_loss = train_loss_accum / dataset.batch_count

    print(f"Epoch {epoch} - Average training loss: {avg_train_loss:.4f}")
