import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torchvision
from ImageAugmentation import SimCLREvalTransform
from torch.utils.data import DataLoader
from sklearn.neighbors import KNeighborsClassifier
import torch.func as fc
from functools import partial


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

    def forward(self, x):
        x = x.detach()
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
             for i in range(num_layers)]
        )
        self.classifier = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes)) if num_classes > 0 else nn.Identity()
        self.posemb = nn.Parameter(torch.rand(1, self.embed.N_patches + 1, d_model))
        self.cls_token = nn.Parameter(torch.rand(1, 1, d_model))
        self.dropout = nn.Dropout(p=dropout)
        self.imgsize = imgsize
        self.patch_dim = patch_dim

    def forward(self, x):
        x = self.embed(x)  # (N,3,H,W) -> (N,H'W',d_model)
        cls_token = self.cls_token.repeat(x.shape[0], 1, 1)
        x = torch.cat([cls_token, x], dim=1)  # (N,num_patches,d_model) -> (N,num_patches+1,d_model)
        x = self.dropout(x + self.posemb)
        for layer in self.encoder:
            x = layer(x)
        x = self.classifier(x[:, 0])
        return x


def AD_func(params, buffers, names, model, x, y):
    z = fc.functional_call(model, ({k: v for k, v in zip(names, params)}, buffers), (x,))
    # ((2N, g) @ (g, 2N)) / (2N,1) @ (1,2N) -> (2N, 2N) / (2N,2N)
    sim_matrix = (z @ z.T) / (z.norm(p=2, dim=1, keepdim=True) @ z.norm(p=2, dim=1, keepdim=True).T)
    mask = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
    pos_mask = mask.roll(shifts=sim_matrix.shape[0]//2, dims=1).bool()  # find pos-pair N away
    pos = torch.exp(sim_matrix[pos_mask] / 0.1)
    neg = torch.exp(sim_matrix.masked_fill(mask, value=float("-inf")) / 0.1)
    loss = -torch.log(pos / torch.sum(neg))
    #loss = - (sim_matrix[pos_mask] / self.hparams.temp / 2) + (torch.logsumexp(sim_matrix.masked_fill(mask, value=float("-inf")) / self.hparams.temp, dim=1) / 2)
    # Find the rank for the positive pair
    sim_matrix = torch.cat([sim_matrix[pos_mask].unsqueeze(1), sim_matrix.masked_fill(pos_mask,float("-inf"))], dim=1)
    pos_pair_pos = torch.argsort(sim_matrix, descending=True, dim=1).argmin(dim=1)
    top1 = torch.mean((pos_pair_pos == 0).float())
    top5 = torch.mean((pos_pair_pos < 5).float())
    mean_pos = torch.mean(pos_pair_pos.float())
    return torch.mean(loss)# , top1, top5, mean_pos


class Trainer:
    def __init__(self,
                 lr,
                 device
                 ):
        self.device = device
        self.model = ViT(num_classes=256, imgsize=32, patch_dim=4, num_layers=6, d_model=256, nhead=4, d_ff_ratio=4, dropout=0.1, activation="relu")
        self.model.to(device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    def train(self, trainloader, valloader, max_epochs):
        for epoch in range(max_epochs):
            train_loss, train_top1, train_top5 = [], [], []
            knn_X_train, knn_y_train = [], []
            self.model.train()

            named_buffers = dict(self.model.named_buffers())
            named_params = dict(self.model.named_parameters())
            names = named_params.keys()
            params = named_params.values()

            for x, y in trainloader:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                v_params = tuple([torch.randn_like(param) for param in params])
                foo = partial(
                    AD_func,
                    model=self.model,
                    names=names,
                    buffers=named_buffers,
                    x=x,
                    y=y
                )

                loss, jvp = fc.jvp(foo, (tuple(params),), (v_params,))
                for v, p in zip(v_params, params):
                    p.grad = v * jvp

                self.optimizer.step()
                out = self.model(x)
                train_loss.append(loss.item())
                #train_top1.append(top1.item())
                #train_top5.append(top5.item())
                knn_X_train.append(out.detach().cpu().numpy())
                knn_y_train.append(y.detach().cpu().numpy())

                #fake_imgs = [img for img in fake[0:3]]
                #grid = torchvision.utils.make_grid(fake_imgs) * 0.5 + 0.5
                #plt.imsave("plot.png", grid.permute(1, 2, 0).detach().cpu().numpy())

            print("Training")
            print("EPOCH: ", epoch)
            print("LOSS: ", np.mean(train_loss))
            #print("TOP1: ", np.mean(train_top1))
            #print("TOP5: ", np.mean(train_top5))
            print("--------------------------------")

            val_loss, val_top1, val_top5 = [], [], []
            knn_X_val, knn_y_val = [], []
            self.model.eval()
            for (x, y) in valloader:
                with torch.no_grad():
                    x, y = x.to(self.device), y.to(self.device)

                    v_params = tuple([torch.randn_like(param) for param in params])
                    foo = partial(
                        AD_func,
                        model=self.model,
                        names=names,
                        buffers=named_buffers,
                        x=x,
                        y=y
                    )

                    loss, jvp = fc.jvp(foo, (tuple(params),), (v_params,))

                    out = self.model(x)
                    val_loss.append(loss.item())
                    #val_top1.append(top1.item())
                    #val_top5.append(top5.item())
                    #import pdb
                    #pdb.set_trace()
                    knn_X_val.append(out.detach().cpu().numpy())
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

    BATCH_SIZE = 256
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

