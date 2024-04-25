from tqdm import tqdm
import torch
from diffusion_model import DiffusionModel

model = DiffusionModel(7, 3)
# load weights
model.load_state_dict(torch.load('diffusion_model.pth'))

noise = torch.randn(1, 7)
condition = torch.tensor([[0.5, 0.2, 1.4]])

for i in tqdm(range(100)):
    x = model(noise, condition)
    noise = x

print(noise)
