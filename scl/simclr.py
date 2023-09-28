from torch import nn
import torch
import lightning as L
import torch.nn.functional as F
import torchvision
from Augmentations import SwavTrainTransform, MNISTTrainTransform
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from ResNet import ResNet
from utils import load_dataset, load_knndataset


class SimCLR(L.LightningModule):
    def __init__(self, imgsize, datasetname, optimizername, output_dim=128, temp=0.1, lr_init=3e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = ResNet(in_channels=3, num_classes=output_dim)
        # KNN validation
        plaintraindataset, plainvaldataset = load_knndataset(name=datasetname, imgsize=imgsize)
        self.plain_trainloader = DataLoader(dataset=plaintraindataset, batch_size=512, shuffle=True, num_workers=20)
        self.plain_valloader = DataLoader(dataset=plainvaldataset, batch_size=512, shuffle=False, num_workers=20)

    def forward(self, x):
        return self.model(x)

    def ntXentLoss(self, xi, xj):
        x = torch.cat([xi, xj], dim=0)
        z = self.forward(x)
        # ((2N, g) @ (g, 2N)) / (2N,1) @ (1,2N) -> (2N, 2N) / (2N,2N)
        sim_matrix = (z @ z.T) / (z.norm(p=2, dim=1, keepdim=True) @ z.norm(p=2, dim=1, keepdim=True).T)
        mask = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
        pos_mask = mask.roll(shifts=sim_matrix.shape[0]//2, dims=1).bool()  # find pos-pair N away
        pos = torch.exp(sim_matrix[pos_mask] / self.hparams.temp)
        neg = torch.exp(sim_matrix.masked_fill(mask, value=float("-inf")) / self.hparams.temp)
        loss = -torch.log(pos / torch.sum(neg))
        #loss = - (sim_matrix[pos_mask] / self.hparams.temp / 2) + (torch.logsumexp(sim_matrix.masked_fill(mask, value=float("-inf")) / self.hparams.temp, dim=1) / 2)
        # Find the rank for the positive pair
        sim_matrix = torch.cat([sim_matrix[pos_mask].unsqueeze(1), sim_matrix.masked_fill(pos_mask,float("-inf"))], dim=1)
        pos_pair_pos = torch.argsort(sim_matrix, descending=True, dim=1).argmin(dim=1)
        top1 = torch.mean((pos_pair_pos == 0).float())
        top5 = torch.mean((pos_pair_pos < 5).float())
        mean_pos = torch.mean(pos_pair_pos.float())
        return torch.mean(loss), top1, top5, mean_pos

    def _shared_step(self, batch, batch_idx, mode="train"):
        images, y = batch
        xi, xj = images
        loss, top1, top5, mean_pos = self.ntXentLoss(xi, xj)
        self.log_dict({
            mode + "_loss": loss,
            mode + "_topp1_acc": top1,
            mode + "_topp5_acc": top5,
            mode + "_mean_acc": mean_pos
        },
            on_step=True,
            on_epoch=True,
            prog_bar=False
        )
        # if batch_idx % 100 == 0:
        #     x_ = zip(xi[:8], xj[:8])
        #     x_ = [xk for xl in x_ for xk in xl]
        #     grid = torchvision.utils.make_grid(torch.stack(x_)[:8], nrow=4)
        #     self.logger.experiment.add_image("cifar-10", grid, self.global_step)
        return {"loss": loss}

    def training_step(self, train_batch, batch_idx):
        return self._shared_step(train_batch, batch_idx, mode="train")

    def validation_step(self, val_batch, batch_idx):
        return self._shared_step(val_batch, batch_idx, mode="val")

    def configure_optimizers(self):
        if self.hparams.optimizername == "sgd":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.hparams.lr_init,
                momentum=0.8
            )
        else:
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.hparams.lr_init,
            )
        # start with initial lr and anneal.
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer,
        #     T_max = self.T,  # max iterations
        #     eta_min=0.0  # min lr
        # )
        # scheduler = {
        #     "scheduler": scheduler,
        #     "interval": "step",
        #     "frequency": 1
        # }
        return [optimizer] #, [scheduler]

    def on_validation_epoch_end(self):
        with torch.no_grad():
            knn = KNeighborsClassifier()
            X_train, X_val, y_train, y_val = [], [], [], []
            for X, y in self.plain_trainloader:
                X_train.append(self.model(X.to(self.device)).detach().cpu().numpy())
                y_train.append(y.detach().numpy())
            for X, y in self.plain_valloader:
                X_val.append(self.model(X.to(self.device)).detach().cpu().numpy())
                y_val.append(y.detach().numpy())
            X_train, y_train = np.concatenate(X_train), np.concatenate(y_train)
            X_val, y_val = np.concatenate(X_val), np.concatenate(y_val)
            knn.fit(X_train, y_train)
            y_hat = knn.predict(X_val)
            acc = np.mean(y_hat == y_val)
            self.log("knn_acc", acc)


if __name__ == "__main__":
    # Dataset
    DATASETNAME = "cifar10"

    # DATA AUGMENTATION
    COLOR_JITTER_STRENGTH = 0.5
    GAUSSIAN_BLUR = False
    IMG_SIZE = 32

    # PARAMS
    TEMP = 0.1

    # HYPERPARAMS
    BATCH_SIZE = 8
    NUM_WORKERS = 20
    MAX_EPOCHS = 50
    OPTIMIZER_NAME = "adam"  # "LARS"
    LR = 3e-4  # 0.075 * BATCH_SIZE ** 0.5

    if DATASETNAME == "cifar10":
        traindataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=True,
            transform=SwavTrainTransform(imgsize=IMG_SIZE, s=COLOR_JITTER_STRENGTH, gaus_blur=GAUSSIAN_BLUR),
            download=True
        )
        valdataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=False,
            transform=SwavTrainTransform(imgsize=IMG_SIZE, s=COLOR_JITTER_STRENGTH, gaus_blur=GAUSSIAN_BLUR),
            download=True
        )
    else:  # "mnist"
        traindataset = torchvision.datasets.MNIST(
            root="./data",
            train=True,
            transform=MNISTTrainTransform(imgsize=IMG_SIZE, s=COLOR_JITTER_STRENGTH, gaus_blur=GAUSSIAN_BLUR),
            download=True
        )
        valdataset = torchvision.datasets.MNIST(
            root="./data",
            train=False,
            transform=MNISTTrainTransform(imgsize=IMG_SIZE, s=COLOR_JITTER_STRENGTH, gaus_blur=GAUSSIAN_BLUR),
            download=True
        )

    trainloader = DataLoader(traindataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valloader = DataLoader(valdataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    torch.set_float32_matmul_precision('medium')

    model = SimCLR(
        imgsize=IMG_SIZE,
        temp=TEMP,
        lr_init=LR,
        datasetname=DATASETNAME,
        optimizername=OPTIMIZER_NAME
    )

    logger = TensorBoardLogger("tb_logs", name="simclr")
    trainer = L.Trainer(
        logger=logger,
        max_epochs=MAX_EPOCHS,
        precision=32,
        accelerator="gpu",
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True,
                mode="min",
                monitor="val_loss",
                save_last=True
            ),
            LearningRateMonitor("epoch")
        ]
    )

    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)

