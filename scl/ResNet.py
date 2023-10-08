import torch
from torch import nn

# Custom ResNets as described in the resnet-paper https://browse.arxiv.org/pdf/1512.03385.pdf


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


class ResNet(nn.Module):
    def __init__(self, blocks, embed_dim, in_channels=3, normlayer=True, first_conv=False, maxpool1=True, expansion=1):
        super().__init__()
        self.expansion = expansion
        in_planes = 64
        # Conv1
        if first_conv:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes) if normlayer else None
        self.relu1 = nn.ReLU()
        if maxpool1:
            self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool1 = nn.MaxPool2d(kernel_size=1, stride=1)
        # ResBlocks
        self.layer1 = self.make_resblock(in_channels=in_planes, out_channels=in_planes, stride=1, padding=1, kernel_size=3, normlayer=normlayer, blocks=blocks[0])
        self.layer2 = self.make_resblock(in_channels=in_planes, out_channels=128, stride=2, padding=1, kernel_size=3, normlayer=normlayer, blocks=blocks[1])
        self.layer3 = self.make_resblock(in_channels=128, out_channels=256, stride=2, padding=1, kernel_size=3, normlayer=normlayer, blocks=blocks[2])
        self.layer4 = self.make_resblock(in_channels=256, out_channels=512, stride=2, padding=1, kernel_size=3, normlayer=normlayer, blocks=blocks[3])
        #
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        fc_layers = [
            nn.Linear(512*self.expansion, 512, bias=False),
            nn.BatchNorm1d(512) if normlayer else None,
            nn.ReLU(),
            nn.Linear(512, embed_dim, bias=True)
        ]
        self.fc = nn.Sequential(*[layer for layer in fc_layers if layer is not None])
        self.normlayer = normlayer

    def make_resblock(self, in_channels, out_channels, kernel_size, padding, stride, blocks, normlayer=True):
        downsample = None
        if stride > 1:
            if normlayer:
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                )
        layers = []
        block = BasicResBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            downsample=downsample,
            normlayer=normlayer
        )
        layers.append(block)
        in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            block = BasicResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                normlayer=normlayer
            )
            layers.append(block)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if self.normlayer:
            x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        # x = self.fc(x)
        return x


class SLCResNet(nn.Module):
    def __init__(self, blocks, embed_dim, in_channels=3, normlayer=True, first_conv=False, maxpool1=True, expansion=1):
        super().__init__()
        self.m = ResNet(blocks=blocks, embed_dim=embed_dim, in_channels=in_channels, normlayer=normlayer, first_conv=first_conv, maxpool1=maxpool1)
        fc_layers = [
            nn.Linear(512*expansion, 512, bias=False),
            nn.BatchNorm1d(512) if normlayer else None,
            nn.ReLU(),
            nn.Linear(512, embed_dim, bias=True)
        ]
        self.q = nn.Sequential(*[layer for layer in fc_layers if layer is not None])

    def forward(self, x):
        x = self.m(x)
        x = self.q(x)
        return x


def resnet18(embed_dim, in_channels=3, normlayer=True, first_conv=False, maxpool1=True):
    return SLCResNet(
        blocks=[2,2,2,2],
        in_channels=in_channels,
        embed_dim=embed_dim,
        normlayer=normlayer,
        first_conv=first_conv,
        maxpool1=maxpool1
    )


def resnet9(embed_dim, in_channels=3, normlayer=True, first_conv=False, maxpool1=True):
    return SLCResNet(
        blocks=[1,1,1,1],
        in_channels=in_channels,
        embed_dim=embed_dim,
        normlayer=normlayer,
        first_conv=first_conv,
        maxpool1=maxpool1
    )


def resnet34(embed_dim, in_channels=3, normlayer=True, first_conv=False, maxpool1=True):
    return SLCResNet(
        blocks=[3,4,6,3],
        in_channels=in_channels,
        embed_dim=embed_dim,
        normlayer=normlayer,
        first_conv=first_conv,
        maxpool1=maxpool1
    )


if __name__ == "__main__":
    model1 = resnet18(256, normlayer=False, maxpool1=False)
    #print(model1)
    model2 = resnet34(256, normlayer=False, maxpool1=False)
    #print(model2)
    x = torch.rand((8, 3, 28, 28))
    out = model1(x)
