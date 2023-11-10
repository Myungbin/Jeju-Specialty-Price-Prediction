import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(self, x, y) -> None:
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        return x.long(), y.long()


class TestDataset(Dataset):
    def __init__(self, x) -> None:
        self.x = torch.from_numpy(x.values)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        return x


def to_numpy(*args):
    return [arg.values for arg in args]
