import torch
from torch import nn
import torchvision.models as models


class ResNettorch(nn.Module):
    # architecture as described in simclr for cifar10
    def __init__(self, modelname, embed_dim, in_channels=3):
        super().__init__()
        resmodels = {
            "resnet18torch": models.resnet18(num_classes=embed_dim),
            "resnet34torch": models.resnet34(num_classes=embed_dim),
            "resnet50torch": models.resnet50(num_classes=embed_dim),
        }
        resmodel = resmodels[modelname]
        resmodel.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone = [
            module for name, module in resmodel.named_children() \
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d)
        ]
        backbone.append(nn.Flatten(start_dim=1))
        self.m = nn.Sequential(*backbone)
        self.q = nn.Sequential(
            nn.Linear(resmodel.fc.in_features, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embed_dim, bias=True)
        )

    def forward(self, x):
        x = self.m(x)
        # x = torch.flatten(x, start_dim=1)
        x = self.q(x)
        return x


class ViTtorch(nn.Module):
    # architecture as described in simclr for cifar10
    def __init__(self, out_dim, in_channels, imgsize, patch_dim, num_layers, d_model, nhead, d_ff_ratio, dropout=0.1, activation="relu"):
        super().__init__()
        self.m = models.VisionTransformer(
            image_size=imgsize,
            patch_size=patch_dim,
            num_layers=num_layers,
            num_heads=nhead,
            hidden_dim=d_model,
            mlp_dim=d_model * d_ff_ratio,
            dropout=dropout,
            attention_dropout=dropout,
            num_classes=out_dim,
        )
        self.m.heads = nn.Identity()
        self.q = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, out_dim))

    def forward(self, x):
        x = self.m(x)
        x = self.q(x)
        return x


if __name__ == "__main__":
    model = ViTtorch(
        out_dim=128,
        in_channels=3,
        imgsize=32,
        patch_dim=4,
        num_layers=7,
        d_model=512,
        nhead=8,
        d_ff_ratio=4,
        dropout=0.1,
        activation="relu"
    )
    print(model)
    x = torch.rand((8, 3, 32, 32))
    out = model(x)
    print(out.shape)