import pandas as pd
import torch.nn as nn
import torch

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from diffusion_model import DiffusionModel

TIME_STEPS = 1000
BETA = 0.99
BATCH_SIZE = 4096
WINDOW_SIZE = 1

X_COLUMNS = ["l0", "l1", "l2", "l3", "l4", "l5", "l6"]
CONDITION_COLUMNS = ["px", "py", "pz"]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("Using GPU")
        return torch.device('cuda')
    else:
        print("Using CPU")
        return torch.device('cpu')


df = pd.read_csv('data.csv')
joints = df[X_COLUMNS].to_numpy()
condition = df[CONDITION_COLUMNS].to_numpy()

data_size = len(X_COLUMNS)
condition_size = len(CONDITION_COLUMNS)
device = get_device()
net = DiffusionModel(device, data_size, condition_size).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01)

joints_train_tensor = torch.tensor(joints).float().to(device)
condition_train_tensor = torch.tensor(condition).float().to(device)
dataset = TensorDataset(joints_train_tensor, condition_train_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

torch.save(net.state_dict(), f'diffusion_model.pth')
for epoch in range(1, 100 + 1):
    for param_group in optimizer.param_groups:
        print("Learning rate:", param_group['lr'])
        break
    for (batch, condition) in tqdm(dataloader):
        optimizer.zero_grad()

        batch_size = batch.size(0)
        step = torch.randint(1, TIME_STEPS, (batch_size, 1)).to(device)
        noise = torch.randn_like(batch).to(device)
        alpha_t = torch.pow(1 - BETA, step)
        alpha_t_last = torch.pow(1 - BETA, torch.max(step - WINDOW_SIZE, torch.zeros_like(step)))
        x = torch.sqrt(alpha_t) * batch + torch.sqrt(1 - alpha_t) * noise
        y = torch.sqrt(alpha_t_last) * batch + torch.sqrt(1 - alpha_t_last) * noise
        x_t = net(x, condition)
        loss = criterion(x_t, y)
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} loss: {loss.item()}")
    if epoch % 10 == 0:
        torch.save(net.state_dict(), f'diffusion_model {epoch}.pth')
