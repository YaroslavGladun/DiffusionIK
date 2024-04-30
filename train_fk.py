import numpy as np
import torch
import torch.nn as nn
from fk import RandomFKDataset, TransformationUtility, JointValuesScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from affine_loss import AffineLoss


# Without batch normalization
# Average test loss for Epoch 33: 0.2286
# Average test rotation loss for Epoch 33: 0.1864
# Average test translation loss for Epoch 33: 0.0422

# With batch normalization
# Average test rotation loss for Epoch 20: 0.1707
# Average test translation loss for Epoch 20: 0.0423
# Learning rate: 0.000920

class Model(nn.Module):
    def __init__(self, device):
        super(Model, self).__init__()
        self.device = device

        self.scaler = JointValuesScaler(device)

        self.fc1 = nn.Linear(14, 128)  # Input layer
        self.bn1 = nn.BatchNorm1d(128)

        self.fc2 = nn.Linear(128, 128)  # Hidden layer 1
        self.bn2 = nn.BatchNorm1d(128)

        self.fc3 = nn.Linear(128, 128)
        self.bn3 = nn.BatchNorm1d(128)

        self.fc4 = nn.Linear(128, 128)
        self.bn4 = nn.BatchNorm1d(128)

        self.fc5 = nn.Linear(128, 128)
        self.bn5 = nn.BatchNorm1d(128)

        self.fc6 = nn.Linear(128, 128)
        self.bn6 = nn.BatchNorm1d(128)

        self.fc7 = nn.Linear(128, 128)
        self.bn7 = nn.BatchNorm1d(128)

        self.fc8 = nn.Linear(128, 128)
        self.bn8 = nn.BatchNorm1d(128)

        self.fc_position = nn.Linear(128, 3)
        self.fc_rotation = nn.Linear(128, 3)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.rpy_multiplier = torch.tensor([np.pi, np.pi / 2, np.pi], device=device)

    def forward(self, x):
        # x = self.scaler(x)
        x_cos = torch.cos(x)
        x_sin = torch.sin(x)
        x = torch.cat((x_cos, x_sin), dim=1)
        x1 = self.bn1(self.relu(self.fc1(x)))
        x2 = self.bn2(self.relu(self.fc2(x1)))
        x3 = self.bn3(self.relu(self.fc3(x2)))
        x4 = self.bn4(self.relu(self.fc4(x3)))
        x5 = self.bn5(self.relu(self.fc5(x4)))
        x6 = self.bn6(self.relu(self.fc6(x5)))
        x7 = self.bn7(self.relu(self.fc7(x6)))
        x8 = self.bn8(self.relu(self.fc8(x7)))
        xyz = self.fc_position(x8)
        rpy = self.fc_rotation(x8)
        # rpy = self.rpy_multiplier * self.tanh(rpy)
        # reshape to 3x3
        # R = R.view(-1, 3, 3)
        # scale rpy to be between (-pi, -pi/2, -pi) and (pi, pi/2, pi)
        # rpy[..., 0] = torch.pi * rpy[..., 0]
        # rpy[..., 1] = torch.pi * rpy[..., 1]
        # rpy[..., 2] = torch.pi * rpy[..., 2]
        x = torch.cat((xyz, rpy), dim=1)

        R, t = TransformationUtility.xyz_rpy_to_torch_affine(x)
        return R, t


# Setup device and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(device).to(device)

# Loss function and optimizer
loss_fn = AffineLoss(alpha=1.0, beta=1.0)
loss_rotation_fn = AffineLoss(alpha=1.0, beta=0.0)
loss_translation_fn = AffineLoss(alpha=0.0, beta=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
lr_space = np.linspace(1e-3, 2 * 1e-4, 200)

# Dataset
train_dataset = RandomFKDataset(device, 10000, 500)
test_dataset = RandomFKDataset(device, 10000, 100)

# Training and testing loops
for epoch in range(1, 200):
    lr = lr_space[epoch - 1]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print(f"Learning rate: {lr:.6f}")
        break
    model.train()  # Set the model to training mode
    train_loss_accum = 0
    test_loss_accum = 0
    test_loss_rotation_accum = 0
    test_loss_translation_accum = 0

    # Training loop
    for i in tqdm(range(train_dataset.batch_count)):
        joints, R, t = train_dataset[i]
        output = model(joints.to(device))
        loss = loss_fn(output, (R, t))

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
            output = model(joints.to(device))
            loss = loss_fn(output, (R, t))
            loss_rotation = loss_rotation_fn(output, (R, t))
            loss_translation = loss_translation_fn(output, (R, t))

            test_loss_accum += loss.item()  # Accumulate the loss
            test_loss_rotation_accum += loss_rotation.item()
            test_loss_translation_accum += loss_translation.item()

    avg_test_loss = test_loss_accum / test_dataset.batch_count
    avg_test_loss_rotation = test_loss_rotation_accum / test_dataset.batch_count
    avg_test_loss_translation = test_loss_translation_accum / test_dataset.batch_count
    print(f"Average test loss for Epoch {epoch}: {avg_test_loss:.4f}")
    print(f"Average test rotation loss for Epoch {epoch}: {avg_test_loss_rotation:.4f}")
    print(f"Average test translation loss for Epoch {epoch}: {avg_test_loss_translation:.4f}")
