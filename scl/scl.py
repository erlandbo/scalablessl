import torch
import lightning as L
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from torch.utils.data import DataLoader
from FeedForward import FeedForward
from ResNet import ResNet
import torchvision
from utils import get_image_stats


class SCL(L.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        if self.hparams.modelarch == "resnet":
            self.model = ResNet(
                in_channels=self.hparams.in_channels,
                embed_dim=self.hparams.embed_dim,
                normlayer=self.hparams.normlayer,
                usemaxpool1=self.hparams.maxpool1
            )
        else:
            self.model = FeedForward(in_channels=3, imgsize=self.hparams.imgsize, out_features=self.hparams.embed_dim)
        # buffer's current values can be loaded using the state_dict of the module which might be useful to know
        self.register_buffer("xi", torch.zeros(1,) + self.hparams.xi)  # weighted sum q
        self.register_buffer("omega", torch.zeros(1,) + self.hparams.omega)  # count q
        # TODO find best sinv_init
        # sinv_init = self.hparams.nsamples** 2 / 10**self.hparams.sinv_init_coeff   # s_init = 10^t * N^-2
        sinv_init = self.hparams.nsamples**self.hparams.ncoeff   # s_init = 10^t * N^-2
        self.register_buffer("s_inv", torch.zeros(1,) + sinv_init)  # scale parameter measure discrepancy between p and q
        self.register_buffer("alpha", torch.zeros(1,) + self.hparams.alpha)  # [0,1] adaptively extra attraction to ease training
        self.register_buffer("ro", torch.zeros(1,) + self.hparams.ro)  # [0,1] forgetting rate s_inv
        self.register_buffer("N", torch.zeros(1,) + self.hparams.nsamples)  # N samples in dataset
        self.T = self.hparams.titer
        self.tau = self.hparams.ncoeff  # s-coefficient N**t

    def forward(self, x):
        return self.model(x)

    def _sim_metric(self, z1, z2):
        if self.hparams.simmetric == "stud-tkernel":
            return 1 / (1 + torch.sum((z1 - z2)**2, dim=1))
        elif self.hparams.simmetric == "gaussian":
            return torch.exp(- torch.sum((z1 - z2)**2, dim=1) / (self.hparams.sigma**2))
        else:  # "cossim"
            z1 = z1.norm(p=2, dim=1, keepdim=True)
            z2 = z2.norm(p=2, dim=1, keepdim=True)
            return torch.sum(z1 * z2, dim=1) * 0.5 + 0.5  # cosine similarity [-1,1] -> [0,1]

    def _shared_step(self, batch, batch_idx, mode="train"):
        # Reset
        self.xi = torch.zeros(1,).to(self.device)
        self.omega = torch.zeros(1,).to(self.device)
        # Process batch
        x_i, xhat_i, x_j = batch
        B = x_i.shape[0]
        z = self.forward(torch.cat([x_i, xhat_i, x_j], dim=0))
        z_i, zhat_i, z_j = z[0:B], z[B:2*B], z[2*B:3*B]
        # Positive forces
        qii = self._sim_metric(z_i, zhat_i)  # (B,1)
        positive_forces = torch.mean( - torch.log(qii) )
        self.xi = self.xi + torch.sum(self.alpha * qii).detach()
        self.omega = self.omega + self.alpha * B
        # Negative forces
        qij = self._sim_metric(z_i, z_j)  # (B,1)
        negative_forces = torch.mean( self.N**self.tau * qij / self.s_inv )
        self.xi = self.xi + torch.sum( (1 - self.alpha) * qij ).detach()
        self.omega = self.omega + (1 - self.alpha) * B
        # Update only in train-step

        loss = positive_forces + negative_forces

        self.log_dict({
            mode + "_loss": loss,
            mode + "_posforces": positive_forces,
            mode + "_negforces": negative_forces,
            mode + "_qii": torch.mean(qii).item(),
            mode + "_qij": torch.mean(qij).item(),
        },
            on_step=True,
            on_epoch=True,
            prog_bar=False
        )
        return {"loss": loss}

    def training_step(self, train_batch, batch_idx):
        self.log_dict({
            "xi": self.xi.item(),
            "omega": self.omega.item(),
            "alpha": self.alpha.item(),
            "ro": self.ro.item(),
            "sinv": self.s_inv.item(),
        },
            on_step=True,
            on_epoch=True,
            prog_bar=False
        )
        loss = self._shared_step(train_batch, batch_idx, mode="train")
        # Update
        self.ro = self.N**self.tau / (self.N**self.tau + self.omega)
        self.s_inv = self.ro * self.s_inv + (1 - self.ro) * self.N**self.tau * self.xi / self.omega
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self._shared_step(val_batch, batch_idx, mode="val")
        if batch_idx % 44 == 0:
            self._show_images(val_batch)

    def configure_optimizers(self):
        if self.hparams.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.hparams.lr,
                momentum=0.8
            )
        else:
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.hparams.lr,
            )
        # start with initial lr and anneal.
        if self.hparams.scheduler:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max = self.T,  # max iterations
                eta_min=0.0  # min lr
            )
            scheduler = {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
            return [optimizer], [scheduler]
        return [optimizer]

    # TODO Better ways to set finetune dataset?
    def load_knn_finetune_dataset(self, traindataset, testdataset):
        self.finetune_trainloader = DataLoader(dataset=traindataset, batch_size=512, shuffle=True, num_workers=self.hparams.numworkers)
        self.finetune_testloader = DataLoader(dataset=testdataset, batch_size=512, shuffle=False, num_workers=self.hparams.numworkers)

    def on_validation_epoch_end(self):
        with torch.no_grad():
            knn = KNeighborsClassifier()
            X_train, X_val, y_train, y_val = [], [], [], []
            for X, y in self.finetune_trainloader:
                X_train.append(self.model(X.to(self.device)).detach().cpu().numpy())
                y_train.append(y.detach().numpy())
            for X, y in self.finetune_testloader:
                X_val.append(self.model(X.to(self.device)).detach().cpu().numpy())
                y_val.append(y.detach().numpy())
            X_train, y_train = np.concatenate(X_train), np.concatenate(y_train)
            X_val, y_val = np.concatenate(X_val), np.concatenate(y_val)
            knn.fit(X_train, y_train)
            y_hat = knn.predict(X_val)
            acc = np.mean(y_hat == y_val)
            self.log("knn_acc", acc)

    def _show_images(self, batch):
        mu, sigma = get_image_stats(self.hparams.dataset)
        mu = torch.tensor(mu, device=self.device)[:, None, None]
        sigma = torch.tensor(sigma, device=self.device)[:, None, None]
        num_imgs = 4
        x_i, xhat_i, x_j = batch
        x_i, xhat_i, x_j = x_i[:num_imgs], xhat_i[:num_imgs], x_j[:num_imgs]
        x_ = zip(x_i, xhat_i, x_j)
        x_ = [xk * sigma + mu for xl in x_ for xk in xl]
        grid = torchvision.utils.make_grid(torch.stack(x_)[:12], nrow=3)
        self.logger.experiment.add_image(self.hparams.dataset, grid, self.global_step)

