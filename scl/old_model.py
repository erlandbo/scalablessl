import torch
from torch import nn


class ResBlock(nn.Module):
    # TODO remove
    def __init__(self, out_channels, downsample=None, kernel_size=3, padding=1, stride=1, in_channels=3):
        super(ResBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ConvBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self,  num_classes, in_channels=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False), #nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.res1 = self.make_resblock(in_channels=64, out_channels=64, stride=1, padding=1, kernel_size=3)
        self.res2 = self.make_resblock(in_channels=64, out_channels=128, stride=2, padding=1, kernel_size=3)
        self.res3 = self.make_resblock(in_channels=128, out_channels=256, stride=2, padding=1, kernel_size=3)
        self.res4 = self.make_resblock(in_channels=256, out_channels=512, stride=2, padding=1, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes, bias=True)
        )

    def make_resblock(self, in_channels, out_channels, kernel_size, padding, stride):
        downsample = None
        if stride > 1:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels)
            )
        return ResBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, downsample=downsample)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        #print(x.shape)
        #x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        # x = self.fc(x)
        return x


class OldResNet(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.m = ResNet(embed_dim)
        self.q = self.fc = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embed_dim, bias=True)
        )

    def forward(self, x):
        x = self.m(x)
        x = self.q(x)
        return x
