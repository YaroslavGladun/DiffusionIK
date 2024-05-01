import torch
import torch.nn as nn

from diffusion_ik import RandomDiffusionIKDataset, Model, DiffusionModel
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

loss_fn = nn.MSELoss()
diffusion_model = DiffusionModel(device, 1000, 1e-4, 0.02)
dataset = RandomDiffusionIKDataset(device, 10000, 2000, diffusion_model)
model = Model(device, diffusion_model, 512).to(device)
optimizer = torch.optim.Adam(model.parameters(), 1e-3)

for epoch in range(100):
    train_loss_accum = 0

    for i in tqdm(range(dataset.batch_count)):
        x, y = dataset[i]

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        if torch.isnan(loss).any():
            print(f"NaN loss at epoch {epoch}, batch {i}")
            # x has nan
            print(f"Input: {torch.isnan(x).any()}")
            # y has nan
            print(f"Output: {torch.isnan(y).any()}")
            print(f"Epoch: {epoch}")
            print(f"Input: {x}")
            print(f"Input: {y}")
            print(f"Output: {y_pred}")
            exit(0)

        train_loss_accum += loss.item()

    avg_train_loss = train_loss_accum / dataset.batch_count

    print(f"Epoch {epoch} - Average training loss: {avg_train_loss:.4f}")
