import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import torchvision
import matplotlib.pyplot as plt
from ImageAugmentation import SimCLREvalTransform, SimCLRTrainTransform
from torch.utils.data import DataLoader
from GAN import GAN2
from PatchDiscriminator import PatchGAN
from VisionTransformer import ViT


class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ViT(0, imgsize=64, patch_dim=4, num_layers=12, d_model=512, nhead=8, d_ff_ratio=4, dropout=0.1, activation="relu")
        self.decoder = GAN2(image_channels=3, latent_dim=512, feature_channels=64)

    def forward(self, x):
        latent = self.encoder(x)
        x_recon = self.decoder(latent[:, :, None, None])
        return x_recon, latent


class Trainer:
    def __init__(self,
                 device,
                 lr,
                 l1lambda
                 ):
        self.device = device
        self.ae = AE()
        self.D = PatchGAN(input_channels=3*2)
        self.ae.to(device)
        self.D.to(device)
        self.opt_G = torch.optim.Adam(self.ae.parameters(), lr=lr, betas=(0.5, 0.999),)
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=lr, betas=(0.5, 0.999),)
        self.L1_LAMBDA = l1lambda

    def train(self, trainloader, valloader, max_epochs):
        for epoch in range(max_epochs):
            D_epoch_real = []
            D_epoch_fake = []
            G_epoch_fake = []
            G_epoch_l1 = []
            knn_X_train, knn_y_train = [], []
            self.ae.train()
            self.D.train()
            for images, labels in trainloader:
                x, y = images
                x, y, labels = x.to(self.device), y.to(self.device), labels.to(self.device)
                # Train D
                y_fake, emb = self.ae(x)
                D_real = self.D(x, y)
                D_real_loss = F.binary_cross_entropy_with_logits(D_real, torch.ones_like(D_real))
                D_fake = self.D(x, y_fake.detach())
                D_fake_loss = F.binary_cross_entropy_with_logits(D_fake, torch.zeros_like(D_fake))
                D_loss = (D_real_loss + D_fake_loss) / 2

                D_epoch_real.append(D_real_loss.item())
                D_epoch_fake.append(D_fake_loss.item())

                self.opt_D.zero_grad()
                D_loss.backward()
                self.opt_D.step()

                # Train G
                D_fake = self.D(x, y_fake)
                G_fake_loss = F.binary_cross_entropy_with_logits(D_fake, torch.ones_like(D_fake))
                L1 = F.l1_loss(y_fake, y) * self.L1_LAMBDA
                G_loss = G_fake_loss + L1

                G_epoch_fake.append(G_fake_loss.item())
                G_epoch_l1.append(L1.item())

                self.opt_G.zero_grad()
                G_loss.backward()
                self.opt_G.step()

            print("D_LOSS_REAL: ", np.mean(D_epoch_real))
            print("D_LOSS_FAKE: ", np.mean(D_epoch_fake))
            print("G_LOSS_FAKE: ", np.mean(G_epoch_fake))
            print("G_L1: ", np.mean(G_epoch_l1))
            print("------------------------------------")

            knn_X_train.append(emb.detach().cpu().numpy())
            knn_y_train.append(labels.detach().cpu().numpy())

            if epoch % 5 == 0:
                real_imgs = [img for img in y[0:3]]
                grid = torchvision.utils.make_grid(real_imgs) * 0.5 + 0.5
                plt.imsave("plot2_real1.png", grid.permute(1, 2, 0).detach().cpu().numpy())

                real_imgs = [img for img in x[0:3]]
                grid = torchvision.utils.make_grid(real_imgs) * 0.5 + 0.5
                plt.imsave("plot2_real2.png", grid.permute(1, 2, 0).detach().cpu().numpy())

                fake_imgs = [img for img in y_fake[0:3]]
                grid = torchvision.utils.make_grid(fake_imgs) * 0.5 + 0.5
                plt.imsave("plot2_fake.png", grid.permute(1, 2, 0).detach().cpu().numpy())

            knn_X_val, knn_y_val = [], []
            self.ae.eval()
            self.D.eval()
            for images, labels in trainloader:
                with torch.no_grad():
                    x, y = images
                    x, y, labels = x.to(self.device), y.to(self.device), labels.to(self.device)
                    y_fake, emb = self.ae(x)
                    knn_X_val.append(emb.detach().cpu().numpy())
                    knn_y_val.append(labels.detach().cpu().numpy())

                    #fake_imgs = [img for img in fake[0:3]]
                    #grid = torchvision.utils.make_grid(fake_imgs) * 0.5 + 0.5
                    #plt.imsave("plot.png", grid.permute(1, 2, 0).detach().cpu().numpy())

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

    IMG_SIZE = 64
    NUM_VIEWS = 2

    BATCH_SIZE = 16
    NUM_WORKERS = 20
    MAX_EPOCHS = 100
    LR = 2e-4
    L1_LAMBDA = 100

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

    trainer = Trainer(device=device, lr=LR, l1lambda=L1_LAMBDA)
    trainer.train(traindataloader, valdataloader, max_epochs=MAX_EPOCHS)
