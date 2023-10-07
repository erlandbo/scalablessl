from torch.utils.data import Dataset
import torch


class SCLDataset(Dataset):
    def __init__(self, basedataset, transform):
        self.basedataset = basedataset
        self.transform = transform

    def __getitem__(self, item):
        # TODO sample other way?
        x, _ = self.basedataset[item]  # uniform [1,...,N]
        x_i = self.transform(x)
        xhat_i = self.transform(x)
        j = torch.randint(low=0, high=len(self.basedataset), size=(1,)).item()
        x_j, _ = self.basedataset[j]  # uniform [1,...,N]^2
        x_j = self.transform(x_j)
        return x_i, xhat_i, x_j

    def __len__(self):
        return len(self.basedataset)


class SCLFinetuneDataset(Dataset):
    def __init__(self, basedataset, transform):
        self.basedataset = basedataset
        self.transform = transform

    def __getitem__(self, item):
        x, y = self.basedataset[item]
        x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.basedataset)


