import pandas as pd
import torch.nn as nn
import torch

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from model import CVAE

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
x_data = df[X_COLUMNS].to_numpy()
condition = df[CONDITION_COLUMNS].to_numpy()


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.mse_loss(x_hat, x)
    KLD = - 0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


input_size = len(X_COLUMNS)
output_size = len(X_COLUMNS)
condition_size = len(CONDITION_COLUMNS)
device = get_device()
net = CVAE(input_size, output_size, 2, condition_size).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

# Convert the data to PyTorch tensors and then to a dataset
# pip install scikit-learn

# Split the data into training and testing sets
x_train, x_test, condition_train, condition_test = train_test_split(
    x_data, condition, test_size=0.1, random_state=42)

# Convert the training data to PyTorch tensors and then to a dataset
x_train_tensor = torch.tensor(x_train).float().to(device)
condition_train_tensor = torch.tensor(condition_train).float().to(device)
train_dataset = TensorDataset(x_train_tensor, condition_train_tensor)

# Convert the testing data to PyTorch tensors and then to a dataset
x_test_tensor = torch.tensor(x_test).float().to(device)
condition_test_tensor = torch.tensor(condition_test).float().to(device)
test_dataset = TensorDataset(x_test_tensor, condition_test_tensor)

train_dataloader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4096, shuffle=True)

# Modify the training loop to handle batches of data
for epoch in range(1, 10000 + 1):
    # print learning rate
    for param_group in optimizer.param_groups:
        print("Learning rate:", param_group['lr'])
        break
    for batch in tqdm(train_dataloader):
        x_batch, condition_batch = batch
        optimizer.zero_grad()
        output, mean, logvar = net(x_batch, condition_batch)
        loss = loss_function(x_batch, output, mean, logvar)
        loss.backward()
        optimizer.step()

    # Testing loop
    test_loss = 0
    with torch.no_grad():
        for x_test, condition_test in test_dataloader:
            test_output, _, _ = net(x_test, condition_test)
            test_loss += criterion(test_output, x_test).item()

    # degrade learning rate
    if epoch % 400 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    # checkpoint
    if epoch % 10 == 0:
        torch.save(net.state_dict(), f"model_{epoch}.pt")

    print(f"Epoch {epoch} loss: {loss.item()}, Test loss: {test_loss / len(test_dataloader)}")
