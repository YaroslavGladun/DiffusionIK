import pandas as pd
import torch.nn as nn
import torch

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

Y_COLUMNS = ["l0", "l1", "l2", "l3", "l4", "l5", "l6"]
# Y_COLUMNS = ["px", "py", "pz", "ox", "oy", "oz", "ow"]
# X_COLUMNS = ["px", "py", "pz"]
X_COLUMNS = ["px", "py", "pz", "l0", "l1", "l2", "l3", "l4", "l5", "l6"]

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
y_data = df[Y_COLUMNS].to_numpy()


class CVAEEncoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CVAEEncoder, self).__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.mean = nn.Linear(256, hidden_size)
        self.logvar = nn.Linear(256, hidden_size)

    def forward(self, x):
        x1 = self.bn1(torch.relu(self.fc1(x)))
        x2 = self.bn2(torch.relu(self.fc2(x1)))
        x3 = self.bn3(torch.relu(self.fc3(x2)))

        mean = self.mean(x3)
        logvar = self.logvar(x3)

        return mean, logvar


class CVAEDecoder(nn.Module):

    def __init__(self, output_size, hidden_size):
        super(CVAEDecoder, self).__init__()

        self.fc1 = nn.Linear(hidden_size, 256)
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc4 = nn.Linear(256, output_size)

    def forward(self, x):
        x1 = self.bn1(torch.relu(self.fc1(x)))
        x2 = self.bn2(torch.relu(self.fc2(x1)))
        x3 = self.bn3(torch.relu(self.fc3(x2)))
        x4 = self.fc4(x3)
        return x4


class CVAE(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, condition_size):
        super(CVAE, self).__init__()
        self.encoder = CVAEEncoder(input_size - condition_size, hidden_size)
        self.decoder = CVAEDecoder(output_size, hidden_size + condition_size)
        self.condition_size = condition_size

    def forward(self, x):
        x, condition = x[:, :-self.condition_size], x[:, -self.condition_size:]
        mean, logvar = self.encoder(x)
        x = self.reparameterization(mean, torch.exp(0.5 * logvar))
        x = torch.cat((x, condition), dim=1)
        x = self.decoder(x)
        return x, mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)
        z = mean + var*epsilon
        return z


def loss_function(x, x_hat, mean, log_var):
    reproduction_loss = nn.functional.mse_loss(x_hat, x)
    KLD = - 0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD


input_size = len(X_COLUMNS)
output_size = len(Y_COLUMNS)
condition_size = len(CONDITION_COLUMNS)
device = get_device()
net = CVAE(input_size, output_size, 64, condition_size).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

# Convert the data to PyTorch tensors and then to a dataset
# pip install scikit-learn
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.1, random_state=42)

# Convert the training data to PyTorch tensors and then to a dataset
x_train_tensor = torch.tensor(x_train).float().to(device)
y_train_tensor = torch.tensor(y_train).float().to(device)
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)

# Convert the testing data to PyTorch tensors and then to a dataset
x_test_tensor = torch.tensor(x_test).float().to(device)
y_test_tensor = torch.tensor(y_test).float().to(device)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_dataloader = DataLoader(train_dataset, batch_size=4096, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4096, shuffle=True)

# Modify the training loop to handle batches of data
for epoch in range(1000):
    # print learning rate
    for param_group in optimizer.param_groups:
        print("Learning rate:", param_group['lr'])
        break
    for batch in tqdm(train_dataloader):
        x_batch, y_batch = batch
        optimizer.zero_grad()
        output, mean, logvar = net(x_batch)
        loss = loss_function(y_batch, output, mean, logvar)
        loss.backward()
        optimizer.step()

    # Testing loop
    test_loss = 0
    with torch.no_grad():
        for x_test, y_test in test_dataloader:
            test_output, _, _ = net(x_test)
            test_loss += criterion(test_output, y_test).item()

    # degrade learning rate
    if epoch % 400 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    print(f"Epoch {epoch} loss: {loss.item()}, Test loss: {test_loss / len(test_dataloader)}")
