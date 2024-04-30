from tqdm import tqdm
import torch
from diffusion_ik import DiffusionModel

device = torch.device('cpu')

model = DiffusionModel(device, 7, 3)
# load weights
model.load_state_dict(torch.load('diffusion_model 50.pth'))

noise = torch.randn(1, 7)
condition = torch.tensor([[0.5, 0.2, 1.4]])

for i in tqdm(range(1000)):
    print(noise)
    x = model(noise, condition)
    noise = x

print(noise)
