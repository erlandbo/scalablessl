import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torchvision
from ImageAugmentation import SimCLREvalTransform
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
from GAN import GAN
import matplotlib.pyplot as plt


class PatchEmbed(nn.Module):
    def __init__(self, imgsize=224, patch_dim=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.N_patches = (imgsize // patch_dim) * (imgsize // patch_dim)
        self.convembed = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_dim, stride=patch_dim)

    def forward(self, x):
        x = self.convembed(x)  # (N,3,H,W) -> (N,embed_dim,H',W')
        x = torch.flatten(x, start_dim=2).transpose(1,2)  #  (N,embed_dim,H',W') -> (N,embed_dim,H'W') -> (N,H'W',embed_dim)
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
        attn = self.W_O(attn.transpose(1,2).contiguous().view(N,L,Ev))  # (N,h,L,dv) -> (N,L,h,dv) -> (N,L,Ev)
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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-5)
        self.logistic_regression = nn.Linear(in_features=d_model,out_features=1)

    def forward(self, x):
        #x = x.detach()
        x_n = self.ln1(x)
        x = x + self.attn(x_n, x_n, x_n)
        x = x + self.mlp(self.ln2(x))
        return self.local_loss(x)

    def local_loss(self, x):
        logits = self.logistic_regression(x[:, 0])
        N = logits.shape[0] // 2
        real, fake = logits[0:N], logits[N:]
        loss_real = F.binary_cross_entropy_with_logits(real, torch.ones_like(real))
        loss_fake = F.binary_cross_entropy_with_logits(fake, torch.zeros_like(fake))
        loss = (loss_fake + loss_real) / 2
        if self.training:
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
        return x, loss.item(), loss_fake


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
        loss = torch.tensor(0.0, device=x.device)
        loss_fake = torch.tensor(0.0, device=x.device)
        for layer in self.encoder:
            x, local_loss, local_loss_fake = layer(x)
            loss += local_loss
            loss_fake += local_loss_fake
        #x = self.classifier(x[:, 0])  # (N,d_model)
        x = x[:, 0]
        return x, loss, loss_fake


class Trainer:
    def __init__(self,
                 lr,
                 device
                 ):
        self.device = device
        self.model = ViT(num_classes=10, imgsize=32, patch_dim=4, num_layers=7, d_model=256, nhead=4, d_ff_ratio=4, dropout=0.1, activation="gelu")
        self.gan = GAN()
        self.model.to(device)
        self.gan.to(device)
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.gan.parameters(), lr=0.0002)

    def train(self, trainloader, valloader, max_epochs):
        for epoch in range(max_epochs):
            train_loss, train_top1, train_top5 = [], [], []
            knn_X_train, knn_y_train = [], []
            self.model.train()
            self.gan.train()
            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                noise = torch.rand((x.shape[0], 100, 1, 1), device=self.device)
                fake = self.gan(noise)
                data = torch.cat([x, fake])
                out, loss, loss_fake = self.model(data)
                self.optimizer.zero_grad()
                loss_fake.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
                #train_top1.append(top1.item())
                #train_top5.append(top5.item())
                knn_X_train.append(out[0:out.shape[0]//2].detach().cpu().numpy())
                knn_y_train.append(y.detach().cpu().numpy())

                fake_imgs = [img for img in fake[0:3]]
                grid = torchvision.utils.make_grid(fake_imgs) * 0.5 + 0.5
                plt.imsave("plot.png", grid.permute(1, 2, 0).detach().cpu().numpy())

            print("Training")
            print("EPOCH: ", epoch)
            print("LOSS: ", np.mean(train_loss))
            #print("TOP1: ", np.mean(train_top1))
            #print("TOP5: ", np.mean(train_top5))
            print("--------------------------------")

            val_loss, val_top1, val_top5 = [], [], []
            knn_X_val, knn_y_val = [], []
            self.model.eval()
            self.gan.eval()
            for (x, y) in valloader:
                with torch.no_grad():
                    x, y = x.to(self.device), y.to(self.device)
                    noise = torch.rand((x.shape[0], 100, 1, 1), device=self.device)
                    fake = self.gan(noise)
                    x = torch.cat([x, fake])
                    out, loss, loss_fake = self.model(x)
                    val_loss.append(loss.item())
                    #val_top1.append(top1.item())
                    #val_top5.append(top5.item())
                    #import pdb
                    #pdb.set_trace()
                    knn_X_val.append(out[0:out.shape[0]//2].detach().cpu().numpy())
                    knn_y_val.append(y.detach().cpu().numpy())
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
    NUM_VIEWS = 1

    BATCH_SIZE = 64
    NUM_WORKERS = 0
    MAX_EPOCHS = 100
    LR = 3e-4

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    traindataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        transform=SimCLREvalTransform(imgsize=IMG_SIZE, num_views=NUM_VIEWS),
        download=True
    )

    valdataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        transform=SimCLREvalTransform(imgsize=IMG_SIZE, num_views=NUM_VIEWS),
        download=True
    )

    traindataloader = DataLoader(traindataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    valdataloader = DataLoader(valdataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    trainer = Trainer(lr=LR, device=device)
    trainer.train(traindataloader, valdataloader, max_epochs=MAX_EPOCHS)

