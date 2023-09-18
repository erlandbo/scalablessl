import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import torchvision
import matplotlib.pyplot as plt
from ImageAugmentation import SimCLREvalTransform, SimCLRTrainTransform
from torch.utils.data import DataLoader

class Interpolate(nn.Module):
    """nn.Module wrapper for F.interpolate."""

    def __init__(self, size=None, scale_factor=None) -> None:
        super().__init__()
        self.size, self.scale_factor = size, scale_factor

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def resize_conv3x3(in_planes, out_planes, scale=1):
    """Upsample + 3x3 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return conv3x3(in_planes, out_planes)
    return nn.Sequential(Interpolate(scale_factor=scale), conv3x3(in_planes, out_planes))


def resize_conv1x1(in_planes, out_planes, scale=1):
    """Upsample + 1x1 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return conv1x1(in_planes, out_planes)
    return nn.Sequential(Interpolate(scale_factor=scale), conv1x1(in_planes, out_planes))


class EncoderBlock(nn.Module):
    """ResNet block, copied from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L35."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class EncoderBottleneck(nn.Module):
    """ResNet bottleneck, copied from
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L75."""

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None) -> None:
        super().__init__()
        width = planes  # this needs to change if we want wide resnets
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class DecoderBlock(nn.Module):
    """ResNet block, but convs replaced with resize convs, and channel increase is in second conv, not first."""

    expansion = 1

    def __init__(self, inplanes, planes, scale=1, upsample=None) -> None:
        super().__init__()
        self.conv1 = resize_conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = resize_conv3x3(inplanes, planes, scale)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        return self.relu(out)


class DecoderBottleneck(nn.Module):
    """ResNet bottleneck, but convs replaced with resize convs."""

    expansion = 4

    def __init__(self, inplanes, planes, scale=1, upsample=None) -> None:
        super().__init__()
        width = planes  # this needs to change if we want wide resnets
        self.conv1 = resize_conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = resize_conv3x3(width, width, scale)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.scale = scale

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        return self.relu(out)


class ResNetEncoder(nn.Module):
    def __init__(self, block, layers, first_conv=False, maxpool1=False) -> None:
        super().__init__()

        self.inplanes = 64
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        if self.first_conv:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        if self.maxpool1:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)


class ResNetDecoder(nn.Module):
    """Resnet in reverse order."""

    def __init__(self, block, layers, latent_dim, input_height, first_conv=False, maxpool1=False) -> None:
        super().__init__()

        self.expansion = block.expansion
        self.inplanes = 512 * block.expansion
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.input_height = input_height

        self.upscale_factor = 8

        self.linear = nn.Linear(latent_dim, self.inplanes * 4 * 4)

        self.layer1 = self._make_layer(block, 256, layers[0], scale=2)
        self.layer2 = self._make_layer(block, 128, layers[1], scale=2)
        self.layer3 = self._make_layer(block, 64, layers[2], scale=2)

        if self.maxpool1:
            self.layer4 = self._make_layer(block, 64, layers[3], scale=2)
            self.upscale_factor *= 2
        else:
            self.layer4 = self._make_layer(block, 64, layers[3])

        if self.first_conv:
            self.upscale = Interpolate(scale_factor=2)
            self.upscale_factor *= 2
        else:
            self.upscale = Interpolate(scale_factor=1)

        # interpolate after linear layer using scale factor
        self.upscale1 = Interpolate(size=input_height // self.upscale_factor)

        self.conv1 = nn.Conv2d(64 * block.expansion, 3, kernel_size=3, stride=1, padding=1, bias=False)

    def _make_layer(self, block, planes, blocks, scale=1):
        upsample = None
        if scale != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                resize_conv1x1(self.inplanes, planes * block.expansion, scale),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, scale, upsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.linear(x)

        # NOTE: replaced this by Linear(in_channels, 514 * 4 * 4)
        # x = F.interpolate(x, scale_factor=4)

        x = x.view(x.size(0), 512 * self.expansion, 4, 4)
        x = self.upscale1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.upscale(x)

        x = self.conv1(x)
        return torch.tanh(x)


def resnet18_encoder(first_conv, maxpool1):
    return ResNetEncoder(EncoderBlock, [2, 2, 2, 2], first_conv, maxpool1)


def resnet18_decoder(latent_dim, input_height, first_conv, maxpool1):
    return ResNetDecoder(DecoderBlock, [2, 2, 2, 2], latent_dim, input_height, first_conv, maxpool1)


def resnet50_encoder(first_conv, maxpool1):
    return ResNetEncoder(EncoderBottleneck, [3, 4, 6, 3], first_conv, maxpool1)


def resnet50_decoder(latent_dim, input_height, first_conv, maxpool1):
    return ResNetDecoder(DecoderBottleneck, [3, 4, 6, 3], latent_dim, input_height, first_conv, maxpool1)


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = resnet18_encoder(first_conv=False, maxpool1=False)
        self.fc = nn.Linear(512, 256)
        self.decoder = resnet18_decoder(latent_dim=256, input_height=32, first_conv=False, maxpool1=False)

    def forward(self, x):
        enc = self.encoder(x)
        latent = self.fc(enc)
        dec = self.decoder(latent)
        return dec, latent


class Trainer:
    def __init__(self,
                 device
                 ):
        self.device = device
        self.model = AE()
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=3e-4)

    def train(self, trainloader, valloader, max_epochs):
        for epoch in range(max_epochs):
            train_loss, train_top1, train_top5 = [], [], []
            knn_X_train, knn_y_train = [], []
            self.model.train()
            for images, y in trainloader:
                xi, xj = images
                x = torch.cat((xi, xj))
                x, y = x.to(self.device), y.to(self.device)
                x_recon, x_latent = self.model(x)
                self.optimizer.zero_grad()
                N = x.shape[0] // 2
                loss = F.mse_loss(x[0:N], x_recon[N:], reduction="sum") + F.mse_loss(x[N:], x_recon[0:N], reduction="sum")
                loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
                #train_top1.append(top1.item())
                #train_top5.append(top5.item())
                knn_X_train.append(x_latent[0:N].detach().cpu().numpy())
                knn_y_train.append(y.detach().cpu().numpy())

                real_imgs = [img for img in x[0:3]]
                grid = torchvision.utils.make_grid(real_imgs) * 0.5 + 0.5
                plt.imsave("plot_real.png", grid.permute(1, 2, 0).detach().cpu().numpy())

                fake_imgs = [img for img in x_recon[N:N+3]]
                grid = torchvision.utils.make_grid(fake_imgs) * 0.5 + 0.5
                plt.imsave("plot_fake.png", grid.permute(1, 2, 0).detach().cpu().numpy())

            print("Training")
            print("EPOCH: ", epoch)
            print("LOSS: ", np.mean(train_loss))
            #print("TOP1: ", np.mean(train_top1))
            #print("TOP5: ", np.mean(train_top5))
            print("--------------------------------")

            val_loss, val_top1, val_top5 = [], [], []
            knn_X_val, knn_y_val = [], []
            self.model.eval()
            for images, y in trainloader:
                with torch.no_grad():
                    xi, xj = images
                    x = torch.cat((xi, xj))
                    x, y = x.to(self.device), y.to(self.device)
                    x_recon, x_latent = self.model(x)
                    N = x.shape[0] // 2
                    loss = F.mse_loss(x[0:N], x_recon[N:], reduction="sum") + F.mse_loss(x[N:], x_recon[0:N], reduction="sum")
                    val_loss.append(loss.item())
                    #train_top1.append(top1.item())
                    #train_top5.append(top5.item())
                    knn_X_val.append(x_latent[0:N].detach().cpu().numpy())
                    knn_y_val.append(y.detach().cpu().numpy())

                    #fake_imgs = [img for img in fake[0:3]]
                    #grid = torchvision.utils.make_grid(fake_imgs) * 0.5 + 0.5
                    #plt.imsave("plot.png", grid.permute(1, 2, 0).detach().cpu().numpy())
            print("Validation")
            print("EPOCH: ", epoch)
            print("LOSS: ", np.mean(val_loss))
            #print("TOP1: ", np.mean(val_top1))
            #print("TOP5: ", np.mean(val_top5))

            knn_val_acc = self.knn_eval(knn_X_train, knn_y_train, knn_X_val, knn_y_val)
            print("KNN VAL ACC: ", knn_val_acc)
            print("--------------------------------")

    @torch.no_grad()
    def knn_eval(self, X_train, y_train, X_val, y_val):
        #self.model.eval()
        knn = KNeighborsClassifier()
        #X_train, X_val, y_train, y_val = [], [], [], []
        #for (X, y) in trainloader:
        #    z, loss = self.model(X.to(self.device))
        #    X_train.append(z.detach().cpu().numpy())
        #    y_train.append(y.detach().numpy())
        #for (X, y) in valloader:
        #    z, loss = self.model(X.to(self.device))
        #    X_val.append(z.detach().cpu().numpy())
        #    y_val.append(y.detach().numpy())
        X_train, y_train = np.concatenate(X_train), np.concatenate(y_train)
        X_val, y_val = np.concatenate(X_val), np.concatenate(y_val)
        knn.fit(X_train, y_train)
        y_hat = knn.predict(X_val)
        acc = np.mean(y_hat == y_val)
        return acc


if __name__ == "__main__":

    IMG_SIZE = 32
    NUM_VIEWS = 2

    BATCH_SIZE = 64
    NUM_WORKERS = 0
    MAX_EPOCHS = 100
    LR = 3e-4

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    traindataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        transform=SimCLRTrainTransform(imgsize=IMG_SIZE, num_views=NUM_VIEWS),
        download=True
    )

    valdataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        transform=SimCLRTrainTransform(imgsize=IMG_SIZE, num_views=NUM_VIEWS),
        download=True
    )

    traindataloader = DataLoader(traindataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    valdataloader = DataLoader(valdataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    trainer = Trainer(device=device)
    trainer.train(traindataloader, valdataloader, max_epochs=MAX_EPOCHS)
