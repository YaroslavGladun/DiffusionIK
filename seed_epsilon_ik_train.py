import numpy as np
import torch
import torch.nn as nn
import wandb

from tqdm import tqdm
from collections import defaultdict
from torch.optim.lr_scheduler import ReduceLROnPlateau

from fk import FK
from common import TransformationUtility
from seed_epsilon_ik_model import SeedEpsilonIKModel
from seed_epsilon_ik_dataset import SeedEpsilonIKDataset
from seed_epsilon_ik_loss import SeedEpsilonIKLoss
from seed_epsilon_ik_config import SeedEpsilonIKConfig
from affine_loss import AffineLoss

config = SeedEpsilonIKConfig()
wandb.init(
    project="ik",
    config=vars(config)
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
model = SeedEpsilonIKModel(device, config).to(device)
# model.load_state_dict(
#     torch.load("/home/yaroslav/Desktop/DiffusionIK/weights/seed_epsilon_ik_5_deg_model_45.pth", map_location=device))

optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=3, threshold=1e-2, threshold_mode="abs", min_lr=1e-6, verbose=True)

dataset = SeedEpsilonIKDataset(device, 8 * 2048, 1000, config)
test_dataset = SeedEpsilonIKDataset(device, 8 * 2048, 1, config)

fk = FK(device)
loss_fn = SeedEpsilonIKLoss(device)
loss_affine = AffineLoss(alpha=1.0, beta=1.0)

epoch = 0
while True:
    model.train()

    train_loss_accum = defaultdict(float)

    test_dataset.max_seed_dist = dataset.max_seed_dist
    print(f"Seed std: {dataset.max_seed_dist:.4f}")

    for i in tqdm(range(dataset.batch_count)):
        pose, seed, epsilon = dataset[i]

        optimizer.zero_grad()
        pred_delta = model(pose, seed, epsilon)
        pose_R, pose_t = pose[:, :9].view(-1, 3, 3), pose[:, 9:].view(-1, 3)
        loss = loss_fn(
            (pose_R, pose_t),
            pred_delta,
            seed
        )
        loss["loss"].backward()
        optimizer.step()

        for key in loss:
            if isinstance(loss[key], torch.Tensor):
                train_loss_accum[key] += loss[key].item()
            else:
                train_loss_accum[key] += loss[key]
    for key in train_loss_accum:
        train_loss_accum[key] /= dataset.batch_count

    for key in train_loss_accum:
        print(f"Epoch {epoch} - Average training {key}: {train_loss_accum[key]:.4f}")

    wandb.log(train_loss_accum)

    # log learning rate of the optimizer
    wandb.log({"lr": optimizer.param_groups[0]["lr"]})
    print(f"Epoch {epoch} - Learning rate: {optimizer.param_groups[0]['lr']:.7f}")

    scheduler.step(train_loss_accum["loss"])

    # model.eval()
    # with torch.no_grad():
    #     pose, seed, epsilon = test_dataset[0]
    #     prediction = model.find_ik(pose, seed, n_times=10)

        # print(f"Epoch {epoch} - Test loss: {loss.item():.4f}")
    epoch += 1

    if epoch % 15 == 0:
        torch.save(model.state_dict(),
                   f"/home/yaroslav/Desktop/DiffusionIK/weights/seed_epsilon_ik_5_deg_model_{epoch}.pth")

    if epoch == 1500:
        break

# torch.save(model.state_dict(), "/content/drive/MyDrive/DiffusionIK/seed_epsilon_ik_model_2048_final.pth")
