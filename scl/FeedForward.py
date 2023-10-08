import torch
from torch import nn


class MLPBlock(nn.Module):
    def __init__(self, in_features, hdim, out_features, activation="relu"):
        super(MLPBlock,self).__init__()
        self.block = nn. Sequential(
            nn.Linear(in_features=in_features, out_features=hdim),
            nn.ReLU() if activation == "relu" else nn.GELU(),
            nn.Linear(in_features=hdim, out_features=out_features),
            nn.ReLU() if activation == "relu" else nn.GELU(),
        )

    def forward(self, x):
        x = self.block(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, imgsize, out_features, in_channels=3):
        super().__init__()
        in_features = imgsize * imgsize * in_channels
        self.block1 = MLPBlock(in_features=in_features, hdim=512, out_features=512)
        self.projector = nn.Sequential(
            nn.Linear(512, out_features)
        )

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.block1(x)
        x = self.projector(x)
        return x
