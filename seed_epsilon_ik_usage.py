import torch
import matplotlib.pyplot as plt

from seed_epsilon_ik_model import SeedEpsilonIKModel
from seed_epsilon_ik_config import SeedEpsilonIKConfig
from seed_epsilon_ik_dataset import SeedEpsilonIKDataset
from affine_loss import AffineLoss
from fk import FK

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = SeedEpsilonIKConfig()
dataset = SeedEpsilonIKDataset(device, 8 * 20480, 1, config)
fk = FK(device)

pose, seed, epsilon = dataset[0]
pose_R, pose_t = pose[:, :9].view(-1, 3, 3), pose[:, 9:].view(-1, 3)
seed_pose_R, seed_pose_t = fk(seed)

affine_loss = AffineLoss(alpha=1.0, beta=1.0)
loss = affine_loss.loss_fn((seed_pose_R, seed_pose_t), (pose_R, pose_t))

epsilon = epsilon.squeeze(dim=-1)

print(loss.shape)
print(epsilon.shape)

plt.scatter(loss.cpu().numpy(), epsilon.cpu().numpy(), s=10, alpha=0.05)
plt.xlabel("Affine loss")
plt.ylabel("Epsilon")
plt.show()