from torch.utils.data import Dataset


class RandomDiffusionIKDataset(Dataset):
    def __init__(self, device):
        self.device = device

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
