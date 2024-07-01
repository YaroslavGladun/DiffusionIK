import torch
import matplotlib.pyplot as plt

from seed_epsilon_ik_model import SeedEpsilonIKModel
from seed_epsilon_ik_config import SeedEpsilonIKConfig
from seed_epsilon_ik_dataset import SeedEpsilonIKDataset
from affine_loss import AffineLoss
from fk import FK

device = torch.device('cpu')
config = SeedEpsilonIKConfig()
model = SeedEpsilonIKModel(device, config).to(device)
model.load_state_dict(torch.load("/home/yaroslav/Desktop/DiffusionIK/weights/seed_epsilon_ik_5_deg_model_15.pth",
                                 map_location=torch.device('cuda')))
model.eval()

dataset = SeedEpsilonIKDataset(device, 1, 1, config)
pose, seed, epsilon = dataset[0]

# estimate inference time
import time
t0 = time.time()
with torch.inference_mode():
    for _ in range(1000):
        model(pose, seed, epsilon)

t1 = time.time()
print(f"Average inference time: {(t1 - t0) / 1000:.4f} s")