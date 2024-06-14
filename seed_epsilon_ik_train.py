import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from fk import FK

from common import TransformationUtility
from seed_epsilon_ik_model import SeedEpsilonIKModel
from seed_epsilon_ik_dataset import SeedEpsilonIKDataset
from seed_epsilon_ik_loss import SeedEpsilonIKLoss
from affine_loss import AffineLoss
from seed_epsilon_ik_config import SeedEpsilonIKConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
config = SeedEpsilonIKConfig()
model = SeedEpsilonIKModel(device, config).to(device)
# model.load_state_dict(torch.load("/home/yaroslav/Desktop/DiffusionIK/weights/seed_epsilon_ik_5_deg_model_15.pth", map_location=device))

# model.load_state_dict(torch.load("seed_epsilon_ik_model.pth", map_location=device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

dataset = SeedEpsilonIKDataset(device, 8 * 2048, 1000, config)
test_dataset = SeedEpsilonIKDataset(device, 8 * 2048, 1, config)

fk = FK(device)
loss_fn = AffineLoss()

lr = 1e-4
epoch = 0
while True:
    model.train()

    train_loss_accum = 0

    test_dataset.max_seed_dist = dataset.max_seed_dist
    print(f"Seed std: {dataset.max_seed_dist:.4f}")
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * 0.997 ** epoch
        print(f"Learning rate: {lr:.6f}")
        break

    for i in tqdm(range(dataset.batch_count)):
        pose, seed = dataset[i]

        optimizer.zero_grad()
        pred_joints = model.forward_autoregressive(pose, seed)
        pred_pose_R, pred_pose_t = fk(pred_joints)
        pose_R, pose_t = pose[:, :9].view(-1, 3, 3), pose[:, 9:].view(-1, 3)
        loss = loss_fn(
            (pred_pose_R, pred_pose_t),
            (pose_R, pose_t)
        )
        loss.backward()
        optimizer.step()

        train_loss_accum += loss.item()

    avg_train_loss = train_loss_accum / dataset.batch_count

    print(f"Epoch {epoch} - Average training loss: {avg_train_loss:.4f}")

    model.eval()
    with torch.no_grad():
        test_loss_accum = 0
        for i in range(test_dataset.batch_count):
            pose, seed = test_dataset[i]
            # pred_joints = model.forward_autoregressive(pose, seed)
            pred_joints = model.forward_autoregressive(pose, seed)
            pred_pose_R, pred_pose_t = fk(pred_joints)
            pose_R, pose_t = pose[:, :9].view(-1, 3, 3), pose[:, 9:].view(-1, 3)
            loss = loss_fn(
                (pred_pose_R, pred_pose_t),
                (pose_R, pose_t)
            )
            test_loss_accum += loss.item()

        avg_test_loss = test_loss_accum / test_dataset.batch_count
        print(f"Epoch {epoch} - Average testing loss: {avg_test_loss:.4f}")

        epoch += 1

    if epoch % 15 == 0:
        torch.save(model.state_dict(), f"/home/yaroslav/Desktop/DiffusionIK/weights/seed_epsilon_ik_5_deg_model_{epoch}.pth")

    if epoch == 1500:
        break

# torch.save(model.state_dict(), "/content/drive/MyDrive/DiffusionIK/seed_epsilon_ik_model_2048_final.pth")


# Epoch 6 - Average training loss: 0.0309
# Epoch 6 - Average testing loss: 0.0288
# Epoch 6 - Average testing affine loss: 0.0288
# Epoch 6 - Average testing seed loss: 0.0446
# Seed std: 0.5236
# Learning rate: 0.000100
