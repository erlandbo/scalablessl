from torch.utils.data import Dataset
import torch


class SCLDataset(Dataset):
    def __init__(self, basedataset, transform):
        self.basedataset = basedataset
        self.transform = transform

    def __getitem__(self, item):
        # TODO improve sampling and clean code sample [1,N]^2
        x, _ = self.basedataset[item]
        x_i = self.transform(x)
        xhat_i = self.transform(x)
        j = torch.randint(low=0, high=len(self.basedataset), size=(1,)).item()
        x_j, _ = self.basedataset[j]
        x_j = self.transform(x_j)
        return x_i, xhat_i, x_j

    def __len__(self):
        return len(self.basedataset)
