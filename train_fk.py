import numpy as np
import torch
import torch.nn as nn
from fk import RandomFKDataset, TransformationUtility
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

# Use ELU activation function
# Average test loss for Epoch 199: 0.0562
# Average test rotation loss for Epoch 199: 0.0438
# Average test translation loss for Epoch 199: 0.0124

# Use SiLU activation function
# Average test loss for Epoch 200: 0.0240
# Average test rotation loss for Epoch 200: 0.0186
# Average test translation loss for Epoch 200: 0.0054

# Use Tanh activation function and atan2 for rotation
# Average test loss for Epoch 156: 0.0157
# Average test rotation loss for Epoch 156: 0.0113
# Average test translation loss for Epoch 156: 0.0044

# With skip connections
# Average test loss for Epoch 184: 0.0127
# Average test rotation loss for Epoch 184: 0.0090
# Average test translation loss for Epoch 184: 0.0037

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

        self.fc_position = nn.Linear(d_model_half, 3)
        self.fc_rotation_x = nn.Linear(d_model_half, 3)
        self.fc_rotation_y = nn.Linear(d_model_half, 3)

        self.activation = nn.SiLU()
        self.sin_cos_activation = nn.Tanh()

    def forward(self, x):
        x_cos = torch.cos(x)
        x_sin = torch.sin(x)
        x = torch.cat((x_cos, x_sin), dim=1)

        x1 = self.bn1(self.activation(self.fc1(x)))

        x2 = self.bn2(self.activation(self.fc2(x1)))

        x3 = self.bn3(self.activation(self.fc3(torch.cat((x1, x2), dim=1))))

        x4 = self.bn4(self.activation(self.fc4(torch.cat((x2, x3), dim=1))))

        x5 = self.bn5(self.activation(self.fc5(torch.cat((x3, x4), dim=1))))

        x6 = self.bn6(self.activation(self.fc6(torch.cat((x4, x5), dim=1))))

        x7 = self.bn7(self.activation(self.fc7(torch.cat((x5, x6), dim=1))))

        x8 = self.bn8(self.activation(self.fc8(torch.cat((x6, x7), dim=1))))

        xyz = self.fc_position(x8)
        rpy_cos = self.sin_cos_activation(self.fc_rotation_x(x8))
        rpy_sin = self.sin_cos_activation(self.fc_rotation_y(x8))
        rpy = torch.atan2(rpy_sin, rpy_cos)
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
lr_space = np.linspace(1e-3, 2 * 1e-5, 200)

# Dataset
train_dataset = RandomFKDataset(device, 10000, 2000)
test_dataset = RandomFKDataset(device, 10000, 100)

# Training and testing loops
for epoch in range(1, len(lr_space) + 1):
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
