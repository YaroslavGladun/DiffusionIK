import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from fk import FK

from probabilistic_ik_model import ProbabilisticIKModel
from probabilistic_ik_dataset import ProbabilisticIKDataset
from affine_loss import AffineLoss
from probabilistic_ik_config import ProbabilisticIKConfig

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
config = ProbabilisticIKConfig()
model = ProbabilisticIKModel(config, device).to(device)
# model.load_state_dict(torch.load("/home/yaroslav/Desktop/DiffusionIK/weights/seed_epsilon_ik_5_deg_model_15.pth", map_location=device))

# model.load_state_dict(torch.load("seed_epsilon_ik_model.pth", map_location=device))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

dataset = ProbabilisticIKDataset(device, 8 * 2048, 1000, config)
test_dataset = ProbabilisticIKDataset(device, 8 * 2048, 1, config)

fk = FK(device)
loss_fn = AffineLoss()

lr = 1e-4
epoch = 0
while True:
    model.train()

    train_loss_accum = 0

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * 0.997 ** epoch
        print(f"Learning rate: {lr:.6f}")
        break

    for i in tqdm(range(dataset.batch_count)):
        optimizer.zero_grad()
        loss = model.forward(dataset[i])
        loss.backward()
        optimizer.step()

        train_loss_accum += loss.item()

    avg_train_loss = train_loss_accum / dataset.batch_count

    print(f"Epoch {epoch} - Average training loss: {avg_train_loss:.4f}")

    model.eval()
    with torch.no_grad():
        test_loss_accum = 0
        for i in range(test_dataset.batch_count):
            target_pose_global = test_dataset[i]["pose"][:, 0, :]
            target_pose_R = target_pose_global[:, :9].view(-1, 3, 3)
            target_pose_t = target_pose_global[:, 9:].view(-1, 3, 1)

            pred_pose_R, pred_pose_t = fk(model.get_ik(test_dataset[i]))
            test_loss_accum += loss_fn((pred_pose_R, pred_pose_t), (target_pose_R, target_pose_t)).item()

        avg_test_loss = test_loss_accum / test_dataset.batch_count
        print(f"Epoch {epoch} - Average testing loss: {avg_test_loss:.4f}")

    epoch += 1

    if epoch % 15 == 0:
        torch.save(
            model.state_dict(),
            f"/home/yaroslav/Desktop/DiffusionIK/weights/seed_epsilon_ik_5_deg_model_{epoch}.pth")

    if epoch == 1500:
        break
