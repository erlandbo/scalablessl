import torch
from torch import nn
from torch.nn import functional as F
import lightning as L
import torchvision
from torch.utils.data import DataLoader
from DinoDataset import DinoAugment, plain_transform
from sklearn.neighbors import KNeighborsClassifier
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import numpy as np
from torchvision import transforms


class PatchEmbed(nn.Module):
    def __init__(self, imgsize=224, patch_dim=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.N_patches = (imgsize // patch_dim) * (imgsize // patch_dim)
        self.convembed = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_dim, stride=patch_dim)

    def forward(self, x):
        x = self.convembed(x)  # (N,3,H,W) -> (N,embed_dim,H',W')
        x = torch.flatten(x, start_dim=2).transpose(1,2)  # (N,embed_dim,H',W') -> (N,embed_dim,H'W') -> (N,H'W',embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.W_Q = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.W_K = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.W_V = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.W_O = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.h = num_heads

    def forward(self, Q, K, V):
        N,L,Ek = Q.shape
        N,S,Ek = K.shape
        N,S,Ev = V.shape
        Q = self.W_Q(Q).view(N, L, self.h, Ek//self.h).transpose(1,2)  # (N,h,L,dk)
        K = self.W_K(K).view(N, S, self.h, Ek//self.h).transpose(1,2)  # (N,h,S,dk)
        V = self.W_V(V).view(N, S, self.h, Ev//self.h).transpose(1,2)  # (N,h,S,dv)
        weights = Q @ K.transpose(-1,-2)   # (N,h,L,dk) @ (N,h,dk,S) -> (N,h,L,S)
        weights = weights * (1 / (Ek/self.h)**0.5)
        weights = F.softmax(weights, dim=-1)
        attn = weights @ V  # (N,h,L,S) @ (N,h,S,dv) -> (N,h,L,dv)
        attn = self.W_O(attn.transpose(1,2).contiguous().view(N,L,Ev)) # (N,h,L,dv) -> (N,L,h,dv) -> (N,L,Ev)
        return attn


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout, activation="relu"):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim=d_model, num_heads=nhead)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        x_n = self.ln1(x)
        x = x + self.attn(x_n, x_n, x_n)
        x = x + self.mlp(self.ln2(x))
        return x


class ViT(nn.Module):
    def __init__(self, num_classes, imgsize, patch_dim, num_layers, d_model, nhead, d_ff_ratio, dropout=0.1, activation="gelu"):
        super().__init__()
        self.embed = PatchEmbed(imgsize=imgsize, patch_dim=patch_dim, embed_dim=d_model)
        self.encoder = nn.ModuleList(
            [EncoderLayer(d_model=d_model, nhead=nhead, d_ff=d_model*d_ff_ratio, dropout=dropout, activation=activation)
             for _ in range(num_layers)]
        )
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes)) if num_classes > 0 else nn.Identity()
        self.cls_token = nn.Parameter(torch.rand(1, 1, d_model))
        self.posemb = nn.Parameter(torch.rand(1, self.embed.N_patches + 1, d_model))
        self.dropout = nn.Dropout(p=dropout)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.embed(x)  # (N,3,H,W) -> (N,H'W',d_model)
        cls_token = self.cls_token.repeat(x.shape[0],1,1)
        x = torch.cat([cls_token, x], dim=1)  # (N,num_patches,d_model) -> (N,num_patches+1,d_model)
        x = self.dropout(x + self.posemb)
        for layer in self.encoder:
            x = layer(x)  # (N,num_patches+1,d_model)
        x = self.classifier(x[:, 0])  # (N,d_model)
        return x


class VisionTransformerModel(L.LightningModule):
    def __init__(self,
                 img_size,
                 patch_dim,
                 d_model,
                 d_ff,
                 nhead,
                 num_layers,
                 num_classes,
                 in_channels,
                 dropout=0.1,
                 lr=1e-3,
                 optimizer_name="sgd",
                 max_epochs=100,
                 ):
        super().__init__()
        self.save_hyperparameters()
        print(img_size, )
        self.vit = ViT(
            num_classes=num_classes,
            imgsize=img_size,
            patch_dim=patch_dim,
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            d_ff_ratio=d_ff,
            dropout=dropout,
            activation="gelu"
        )
        self.CE = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.vit.forward(x)
        return x

    def _shared_step(self, batch, mode="train"):
        x, y = batch
        logits = self.forward(x)
        loss = self.CE(logits, y)
        acc = torch.mean((torch.argmax(logits, dim=1) == y).float())
        self.log_dict({
            mode + "_loss": loss,
            mode + "_acc": acc
        },
            on_step=True,
            on_epoch=True,
            prog_bar=False
        )
        return {"loss": loss, "acc": acc}

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, mode="val")

    def configure_optimizers(self):
        if self.hparams.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(self.vit.parameters(), lr=self.hparams.lr)
            return [optimizer]
        else:
            optimizer = torch.optim.Adam(self.vit.parameters(), lr=self.hparams.lr)
        return [optimizer]


if __name__ == "__main__":
    # DATASET
    IMG_SIZE = 32
    IN_CHANNELS = 3

    # MODEL
    PATCH_DIM = 4
    N_PATCHES = IMG_SIZE ** 2 // PATCH_DIM ** 2
    D_MODEL = 512  # embed dim
    D_FF = 4
    N_HEAD = 8
    NUM_LAYERS = 6
    NUM_CLASSES = 10
    DROPOUT = 0.1

    # HYPERPARAMS
    BATCH_SIZE = 64
    NUM_WORKERS = 20
    MAX_EPOCHS = 500
    OPTIMIZER_NAME = "adamw"  # "LARS"
    LR = 3e-4  # 0.075 * BATCH_SIZE ** 0.5
    WEIGHT_DECAY = 1e-4

    torch.set_float32_matmul_precision('medium')  # | 'high')

    model = VisionTransformerModel(
        img_size=IMG_SIZE,
        patch_dim=PATCH_DIM,
        d_model=D_MODEL,
        d_ff=D_FF,
        nhead=N_HEAD,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        dropout=DROPOUT,
        lr=LR,
        optimizer_name=OPTIMIZER_NAME,
        max_epochs=MAX_EPOCHS,
        in_channels = IN_CHANNELS
    )

    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ])

    traindataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        transform=transform,
        download=True
    )

    valdataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        transform=transform,
        download=True
    )

    trainloader = DataLoader(traindataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
    valloader = DataLoader(valdataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False)

    logger = TensorBoardLogger("tb_logs", name="ViT-Cifar10")
    trainer = L.Trainer(
        logger=logger,
        max_epochs=MAX_EPOCHS,
        precision=16,
        accelerator="gpu",
    )
    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)