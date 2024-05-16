import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from fk import FK

from common import TransformationUtility
from seed_epsilon_ik_model import SeedEpsilonIKModel
from seed_epsilon_ik_dataset import SeedEpsilonIKDataset
from affine_loss import AffineLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
model = SeedEpsilonIKModel(2048).to(device)
# model.load_state_dict(torch.load("seed_epsilon_ik_model.pth", map_location=device))
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

lr_space = np.linspace(1e-3, 2 * 1e-5, 500)

dataset = SeedEpsilonIKDataset(device, 4 * 2048, 1)
test_dataset = SeedEpsilonIKDataset(device, 2048, 1)

fk = FK(device)
affine_loss = AffineLoss(alpha=1.0, beta=1.0)

for epoch in range(len(lr_space)):
    model.train()

    train_loss_accum = 0

    lr = lr_space[epoch]
    lr = 1e-5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print(f"Learning rate: {lr:.6f}")
        break

    for i in tqdm(range(dataset.batch_count)):
        pose, seed, epsilon = dataset[i]

        optimizer.zero_grad()
        pred_joints = model(pose, seed, epsilon)
        pred_pose_R, pred_pose_t = fk(pred_joints)
        pose_R, pose_t = pose[:, :9].view(-1, 3, 3), pose[:, 9:].view(-1, 3)
        loss = affine_loss((pose_R, pose_t), (pred_pose_R, pred_pose_t))
        loss.backward()
        optimizer.step()

        train_loss_accum += loss.item()

    avg_train_loss = train_loss_accum / dataset.batch_count

    print(f"Epoch {epoch} - Average training loss: {avg_train_loss:.4f}")

    model.eval()
    with torch.no_grad():
        test_loss_accum = 0
        for i in range(test_dataset.batch_count):
            pose, seed, epsilon = test_dataset[i]
            pred_joints = model(pose, seed, epsilon)
            pred_pose_R, pred_pose_t = fk(pred_joints)
            pose_R, pose_t = pose[:, :9].view(-1, 3, 3), pose[:, 9:].view(-1, 3)
            loss = affine_loss((pose_R, pose_t), (pred_pose_R, pred_pose_t))
            test_loss_accum += loss.item()

        avg_test_loss = test_loss_accum / test_dataset.batch_count
        print(f"Epoch {epoch} - Average testing loss: {avg_test_loss:.4f}")
