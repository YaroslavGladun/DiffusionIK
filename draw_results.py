import torch
import matplotlib.pyplot as plt

from seed_epsilon_ik_model import SeedEpsilonIKModel
from seed_epsilon_ik_config import SeedEpsilonIKConfig
from seed_epsilon_ik_dataset import SeedEpsilonIKDataset
from seed_epsilon_ik_loss import SeedEpsilonIKLoss

device = torch.device("cpu")
config = SeedEpsilonIKConfig()
model = SeedEpsilonIKModel(device, config)
model.load_state_dict(
    torch.load("/home/yaroslav/Desktop/DiffusionIK/weights/seed_epsilon_ik_5_deg_model_15.pth", map_location=device))
model.eval()

loss_fn = SeedEpsilonIKLoss(device)

dataset = SeedEpsilonIKDataset(device, 1, 1, config)
pose, seed, _ = dataset[0]
epsilon = torch.linspace(0, config.max_seed_dist, 1000).to(device).view(-1, 1)
pose = pose.view(1, -1).repeat(1000, 1)
seed = seed.view(1, -1).repeat(1000, 1)

pred_delta = model(pose, seed, epsilon)
pose_R, pose_t = pose[:, :9].view(-1, 3, 3), pose[:, 9:].view(-1, 3)

losses = []

for i in range(1000):
    loss = loss_fn(
        (pose_R[i].unsqueeze(0), pose_t[i].unsqueeze(0)),
        pred_delta[i].view(1, -1),
        seed[i].view(1, -1)
    )
    losses.append(loss["cartesian_loss"].item())

plt.plot(epsilon.cpu().numpy(), losses)
plt.xlabel("Epsilon")
plt.ylabel("Cartesian loss")
plt.xlim(0, config.max_seed_dist)
plt.ylim(0, 1)
plt.show()