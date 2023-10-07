import torch
from torch import nn


class BasicResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, kernel_size=3, padding=1, normlayer=True):
        super(BasicResBlock,self).__init__()
        self.downsample = downsample
        self.normlayer = normlayer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.batchnorm1 = nn.BatchNorm2d(out_channels) if normlayer else None
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.batchnorm2 = nn.BatchNorm2d(out_channels) if normlayer else None
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        if self.normlayer:
            out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.normlayer:
            out = self.batchnorm2(out)
        if self.downsample:
            identity = self.downsample(identity)
        out += identity
        out = self.relu(out)
        return out


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
    # TODO add more support and improve
    def __init__(self, in_channels, embed_dim, normlayer=True, usemaxpool1=True, first_conv=False):
        super().__init__()
        # Conv1
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64) if normlayer else None
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if usemaxpool1 else None
        # ResBlocks
        self.res1 = self.make_resblock(in_channels=64, out_channels=64, stride=1, padding=1, kernel_size=3, normlayer=normlayer)
        self.res2 = self.make_resblock(in_channels=64, out_channels=128, stride=2, padding=1, kernel_size=3, normlayer=normlayer)
        self.res3 = self.make_resblock(in_channels=128, out_channels=256, stride=2, padding=1, kernel_size=3, normlayer=normlayer)
        self.res4 = self.make_resblock(in_channels=256, out_channels=512, stride=2, padding=1, kernel_size=3, normlayer=normlayer)
        #
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        fc_layers = [
            nn.Linear(512*4, 512, bias=False),
            nn.BatchNorm1d(512) if normlayer else None,
            nn.ReLU(),
            nn.Linear(512, embed_dim, bias=True)
        ]
        self.fc = nn.Sequential(*[layer for layer in fc_layers if layer is not None])
        self.normlayer = normlayer
        self.usemaxpool1 = usemaxpool1

    def make_resblock(self, in_channels, out_channels, kernel_size, padding, stride, normlayer=True):
        downsample = None
        if stride > 1:
            if normlayer:
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                )
        block = BasicResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            downsample=downsample,
            normlayer=normlayer
        )
        return block

    def forward(self, x):
        x = self.conv1(x)
        if self.normlayer: x = self.bn1(x)
        x = self.relu1(x)
        if self.usemaxpool1: x = self.maxpool1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        # x = self.pool(x)
        print(x.shape)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    model = ResNet(in_channels=3, embed_dim=128, normlayer=True)
    print(model)
    x = torch.rand((8, 3, 28, 28))
    out = model(x)
    print(out.shape)