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


class SwAV(L.LightningModule):
    def __init__(self, optimizername,batchsize, imgsize, datasetname, output_dim=128, num_prototypes=100, temp=0.1, lr_init=3e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = ResNet(in_channels=3, num_classes=output_dim)
        self.prototypes = nn.Linear(in_features=output_dim, out_features=num_prototypes)
        # KNN validation
        plaintraindataset, plainvaldataset = load_knndataset(name=datasetname, imgsize=imgsize)
        self.plain_trainloader = DataLoader(dataset=plaintraindataset, batch_size=512, shuffle=True, num_workers=20)
        self.plain_valloader = DataLoader(dataset=plainvaldataset, batch_size=512, shuffle=False, num_workers=20)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx, mode="train"):
        images, y = batch
        with torch.no_grad():
            w = model.prototypes.weight.data.clone()
            w = F.normalize(w, dim=1, p=2)
            model.prototypes.weight.copy_(w)
        xt, xs = images
        z = self.forward(torch.cat([xt, xs], dim=0))
        z = F.normalize(z, p=2, dim=1)
        B = z.shape[0] // 2
        scores = self.prototypes(z)  # (2B,D) @ (D,K) -> (2B,K)
        scores_t = scores[:B]
        scores_s = scores[B:]
        with torch.no_grad():
            q_t = self.sinkhorn(scores_t)  # (B,K)
            q_s = self.sinkhorn(scores_s)
        p_t = F.softmax(scores_t / self.hparams.temp, dim=1)
        p_s = F.softmax(scores_s / self.hparams.temp, dim=1)
        #loss = -0.5 * torch.mean(torch.sum(q_t * torch.log(p_s) + q_s * torch.log(p_t), dim=1))  # (B,K)*(B,K) -> (B,1) -> (1,)
        #loss = -0.5 * torch.mean(q_t * torch.log(p_s) + q_s * torch.log(p_t))  # (B,K)*(B,K) -> (B,1) -> (1,)
        loss = torch.mean(-0.5 * torch.sum(q_t * torch.log(p_s), dim=1) + -0.5*torch.sum(q_s * torch.log(p_t), dim=1))
        self.log_dict({
            mode + "_loss": loss,
        },
            on_step=True,
            on_epoch=True,
            prog_bar=False
        )
        return {"loss": loss}

    def training_step(self, train_batch, batch_idx):
        return self._shared_step(train_batch, batch_idx, mode="train")

    def validation_step(self, val_batch, batch_idx):
        return self._shared_step(val_batch, batch_idx, mode="val")

    def on_after_backward(self):
        if self.current_epoch < 10:
            for name, param in self.model.named_parameters():
                if "prototypes" in name:
                    param.grad = None

    def sinkhorn(self, scores, eps=0.05, niters=3):
        B, K = scores.shape
        Q = torch.exp(scores / eps).T  # (B,K) -> (K,B)
        Q = Q / torch.sum(Q)
        u, r, c = torch.zeros(K, device=self.device), torch.ones(K, device=self.device)/K, torch.ones(B, device=self.device)/B
        for _ in range(niters):
            u = torch.sum(Q, dim=1)
            Q = Q * (r / u).unsqueeze(1)
            Q = Q * (c / torch.sum(Q,dim=0)).unsqueeze(0)
        return (Q / torch.sum(Q,dim=0, keepdim=True)).T  # ((K,B) / (1,B)).T -> ((K,B)/(K,B)).T -> (B,K)

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

    # Datset
    DATASETNAME = "cifar10"

    # DATA AUGMENTATION
    COLOR_JITTER_STRENGTH = 0.5
    GAUSSIAN_BLUR = False
    IMG_SIZE = 32

    # PARAMS
    TEMP = 0.1
    NUM_PROTOTYPES = 3000

    # HYPERPARAMS
    BATCH_SIZE = 8
    NUM_WORKERS = 20
    MAX_EPOCHS = 200
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

    model = SwAV(
        batchsize=BATCH_SIZE,
        imgsize=IMG_SIZE,
        temp=TEMP,
        num_prototypes=NUM_PROTOTYPES,
        lr_init=LR,
        datasetname=DATASETNAME,
        optimizername=OPTIMIZER_NAME
    )

    logger = TensorBoardLogger("tb_logs", name="others")
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

