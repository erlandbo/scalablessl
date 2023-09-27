import torch
import lightning as L
from Datasets import SCLDataset
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from Augmentations import SwavTrainTransform
from utils import load_dataset, load_knndataset
from torch.utils.data import DataLoader, RandomSampler
from FeedForward import FeedForward
from ResNet import ResNet


class SCL(L.LightningModule):
    def __init__(self,
                 s_inv,
                 xi,
                 omega,
                 ro,
                 alpha,
                 imgsize,
                 N_samples,
                 simmetric,
                 T_iterations,
                 embed_dim,
                 lr_init,
                 datasetname,
                 modelname,
                 ):
        super().__init__()
        self.save_hyperparameters()
        if modelname == "resnet":
            self.model = ResNet(in_channels=3, num_classes=embed_dim)
        else:
            self.model = FeedForward(in_channels=3, imgsize=imgsize, out_features=embed_dim)
        # buffer's current values can be loaded using the state_dict of the module which might be useful to know
        self.register_buffer("xi", torch.zeros(1,) + xi)  # weighted sum q
        self.register_buffer("omega", torch.zeros(1,) + omega)  # count q
        self.register_buffer("s_inv", torch.zeros(1,) + s_inv)  # scale parameter measure discrepancy between p and q
        self.register_buffer("alpha", torch.zeros(1,) + alpha)  # [0,1] adaptively extra attraction to ease training
        self.register_buffer("ro", torch.ones(1,) + ro)  # [0,1] forgetting rate s_inv
        self.register_buffer("N", torch.zeros(1,) + N_samples)  # N samples in dataset
        self.T = T_iterations
        # KNN validation
        plaintraindataset, plainvaldataset = load_knndataset(name=datasetname, imgsize=imgsize)
        self.plain_trainloader = DataLoader(dataset=plaintraindataset, batch_size=512, shuffle=True, num_workers=20)
        self.plain_valloader = DataLoader(dataset=plainvaldataset, batch_size=512, shuffle=False, num_workers=20)

    def forward(self, x):
        return self.model(x)

    def _sim_metric(self, z1, z2):
        if self.hparams.simmetric == "stud-tkernel":
            return 1 / (1 + torch.sum((z1 - z2)**2, dim=1))
        elif self.hparams.simmetric == "l2":
            return 1 / (torch.sum((z1 - z2)**2, dim=1))  # fix div /0
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
        # positive forces
        qii = self._sim_metric(z_i, zhat_i)
        qii = torch.mean(qii)
        positive_forces = - torch.log(qii)
        self.xi = self.xi + self.alpha * qii.detach()
        self.omega = self.omega + self.alpha
        # negative forces
        qij = self._sim_metric(z_i, z_j)
        qij = torch.mean(qij)
        negative_forces = qij / self.s_inv
        self.xi = self.xi + (1 - self.alpha) * qij.detach()
        self.omega = self.omega + (1 - self.alpha)

        # Update
        self.omega = self.omega * 128**2
        self.xi = self.xi * 128**2
        self.ro = self.N / (self.N + self.omega)
        self.s_inv = self.ro * self.s_inv + (1 - self.ro) * self.N * self.xi / self.omega

        loss = positive_forces + negative_forces

        self.log_dict({
            mode + "_loss": loss,
            mode + "_posforces": positive_forces,
            mode + "_negforces": negative_forces,
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
            on_epoch=False,
            prog_bar=False
        )
        return self._shared_step(train_batch, batch_idx, mode="train")

    def validation_step(self, val_batch, batch_idx):
        return self._shared_step(val_batch, batch_idx, mode="val")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.lr_init,
        )
        # start with initial lr, cosine anneal to 0 then restart.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max = self.T,  # max iterations
            eta_min=0.0
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [scheduler]

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
    # DATASET
    DATASET = "mnist"

    # DATA AUGMENTATION
    COLOR_JITTER_STRENGTH = 0.5
    GAUSSIAN_BLUR = False
    IMG_SIZE = 32

    # HYPERPARAMS
    BATCH_SIZE = 8
    NUM_WORKERS = 20
    OPTIMIZER_NAME = "sgd"
    LR_INIT = 1.0  # lr?

    traindataset, valdataset = load_dataset(DATASET)

    traindataset = SCLDataset(traindataset,
                              transform=SwavTrainTransform(imgsize=IMG_SIZE, s=COLOR_JITTER_STRENGTH, gaus_blur=GAUSSIAN_BLUR, num_views=1, dataset=DATASET),
                              )
    valdataset = SCLDataset(valdataset,
                            transform=SwavTrainTransform(imgsize=IMG_SIZE, s=COLOR_JITTER_STRENGTH, gaus_blur=GAUSSIAN_BLUR, num_views=1, dataset=DATASET),
                            )

    # NOTE: sampling with replacement
    trainloader = DataLoader(traindataset,
                             batch_size=BATCH_SIZE,
                             num_workers=NUM_WORKERS,
                             sampler=RandomSampler(traindataset, replacement=True)  # Sample randomly with replacement
                             )
    valloader = DataLoader(valdataset,
                           batch_size=BATCH_SIZE,
                           num_workers=NUM_WORKERS,
                           sampler=RandomSampler(valdataset, replacement=True)
                           )

    # PARAMS
    MODELNAME = "feedforward"
    N_SAMPLES = len(traindataset)
    ALPHA = 0.5
    T_ITER = 100_000
    RO = 1.0
    XI = 0.0
    OMEGA = 0.0
    S_INV = N_SAMPLES
    EMBED_DIM = 128
    SIMMETRIC = "stud-tkernel"  # "cossim", "l2", "stud-tkernel"

    MAX_EPOCHS = -1
    # MAX_STEPS = T_ITER

    model = SCL(
        imgsize=IMG_SIZE,
        lr_init=LR_INIT,
        N_samples=len(traindataset),
        T_iterations=T_ITER,
        datasetname=DATASET,
        modelname=MODELNAME,
        alpha=ALPHA,
        ro=RO,
        xi=XI,
        omega=OMEGA,
        s_inv=S_INV,
        simmetric=SIMMETRIC,
        embed_dim=EMBED_DIM
    )

    # Lightning
    torch.set_float32_matmul_precision('medium')

    logger = TensorBoardLogger("tb_logs", name="scl")
    trainer = L.Trainer(
        logger=logger,
        max_epochs=MAX_EPOCHS,
        max_steps=T_ITER,
        precision=32,
        accelerator="gpu",
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True,
                mode="min",
                monitor="val_loss",
                save_last=True
            ),
            LearningRateMonitor("step")
        ]
    )

    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)

