from torch import nn
import torch


class GanBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, last_layer=False):
        super().__init__()
        if not last_layer:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Tanh()
            )

    def forward(self, x):
        return self.block(x)


class GAN2(nn.Module):
    def __init__(self, image_channels=3, latent_dim=100, feature_channels=64):
        super().__init__()
        self.block1 = GanBlock(latent_dim, 1024)
        self.block2 = GanBlock(1024, 512)
        self.block3 = GanBlock(512, 512)
        self.block4 = GanBlock(512, 256)
        self.block5 = GanBlock(256, 64)
        self.last_block = GanBlock(64, image_channels, last_layer=True)

    def forward(self, x):
        x = self.block1(x)  # (N,latent,1,1) -> (N,feature_channels * 8, 2, 2)
        x = self.block2(x)  # (N,feature_channels * 8, 2, 2) -> (N,feature_channels * 4, 4, 4)
        x = self.block3(x)  # (N,feature_channels * 4, 4, 4) -> (N,feature_channels * 2, 8, 8)
        x = self.block4(x)  # (N,feature_channels * 2, 8, 8) -> (N,feature_channels, 16, 16)
        x = self.block5(x)  # (N,feature_channels * 2, 8, 8) -> (N,feature_channels, 32, 32)
        x = self.last_block(x)  # (N,feature_channels, 16, 16) -> (N,image_channels, 64, 64)
        return x


class GAN(nn.Module):
    def __init__(self, image_channels=3, latent_dim=100, feature_channels=64):
        super().__init__()
        self.block1 = GanBlock(latent_dim, feature_channels * 8)
        self.block2 = GanBlock(feature_channels*8, feature_channels*4)
        self.block3 = GanBlock(feature_channels*4, feature_channels*2)
        self.block4 = GanBlock(feature_channels*2, feature_channels)
        self.last_block = GanBlock(feature_channels, image_channels, last_layer=True)

    def forward(self, x):
        x = self.block1(x)  # (N,latent,1,1) -> (N,feature_channels * 8, 2, 2)
        x = self.block2(x)  # (N,feature_channels * 8, 2, 2) -> (N,feature_channels * 4, 4, 4)
        x = self.block3(x)  # (N,feature_channels * 4, 4, 4) -> (N,feature_channels * 2, 8, 8)
        x = self.block4(x)  # (N,feature_channels * 2, 8, 8) -> (N,feature_channels, 16, 16)
        x = self.last_block(x)  # (N,feature_channels, 16, 16) -> (N,image_channels, 32, 32)
        return x


if __name__ == "__main__":
    x = torch.rand(64, 100, 1, 1)
    model = GAN()
    out = model(x)
    print(out.shape)
