import torch
import torch.nn as nn
from fk import RandomFKDataset, TransformationUtility
from torch.utils.data import DataLoader
from tqdm import tqdm


class Model(nn.Module):
    def __init__(self, device):
        super(Model, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(7, 256)  # Input layer
        self.fc2 = nn.Linear(256, 256)  # Hidden layer 1
        self.fc3 = nn.Linear(256, 256)  # Hidden layer 2
        self.fc4 = nn.Linear(256, 256)  # Hidden layer 3
        self.fc5 = nn.Linear(256, 256)  # Hidden layer 4
        self.fc_xyz = nn.Linear(256, 3)
        self.fc_rpy = nn.Linear(256, 3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        xyz = self.fc_xyz(x)
        rpy = torch.pi * self.tanh(self.fc_rpy(x))
        # scale rpy to be between (-pi, -pi/2, -pi) and (pi, pi/2, pi)
        # rpy[..., 0] = torch.pi * rpy[..., 0]
        # rpy[..., 1] = torch.pi * rpy[..., 1]
        # rpy[..., 2] = torch.pi * rpy[..., 2]
        x = torch.cat((xyz, rpy), dim=1)

        R, t = TransformationUtility.xyz_rpy_to_torch_affine(x)
        R, t = R.view(-1, 9), t.view(-1, 3)
        x = torch.cat((R, t), dim=1)
        return x


# Setup device and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(device).to(device)

# Loss function and optimizer
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Dataset
train_dataset = RandomFKDataset(device, 2048, 500)
test_dataset = RandomFKDataset(device, 2048, 100)

# Training and testing loops
for epoch in range(1, 200):
    model.train()  # Set the model to training mode
    train_loss_accum = 0
    test_loss_accum = 0

    # Training loop
    for i in tqdm(range(train_dataset.batch_count)):
        joints, R, t = train_dataset[i]
        R, t = R.reshape(-1, 9), t.reshape(-1, 3)
        target = torch.cat((R, t), dim=1)
        output = model(joints.to(device))
        loss = loss_fn(output, target.to(device))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_accum += loss.item()  # Accumulate the loss

    avg_train_loss = train_loss_accum / train_dataset.batch_count
    print(f"Average training loss for Epoch {epoch}: {avg_train_loss:.4f}")

    # Testing loop
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient computation during evaluation
        for i in tqdm(range(test_dataset.batch_count)):
            joints, R, t = test_dataset[i]
            R, t = R.reshape(-1, 9), t.reshape(-1, 3)
            target = torch.cat((R, t), dim=1)
            output = model(joints.to(device))
            loss = loss_fn(output, target.to(device))

            test_loss_accum += loss.item()  # Accumulate the loss

    avg_test_loss = test_loss_accum / test_dataset.batch_count
    print(f"Average test loss for Epoch {epoch}: {avg_test_loss:.4f}")
