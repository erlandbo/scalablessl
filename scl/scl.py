from torch import nn
import torch
from ConvLayers import ResBlock
import lightning as L
import torchvision
from torch.utils.data import DataLoader, RandomSampler
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from Augmentations import SwavTrainTransform, SwavEvalTransform


class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False), #nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.res1 = self.make_resblock(in_channels=64, out_channels=64, stride=1, padding=1, kernel_size=3)
        self.res2 = self.make_resblock(in_channels=64, out_channels=128, stride=2, padding=1, kernel_size=3)
        self.res3 = self.make_resblock(in_channels=128, out_channels=256, stride=2, padding=1, kernel_size=3)
        self.res4 = self.make_resblock(in_channels=256, out_channels=512, stride=2, padding=1, kernel_size=3)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(2048, 512, bias=False),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes, bias=True)
        )

    def make_resblock(self, in_channels, out_channels, kernel_size, padding, stride):
        downsample = None
        if stride > 1:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
                nn.BatchNorm2d(out_channels)
            )
        return ResBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, downsample=downsample)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        #print(x.shape)
        #x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


class SCL(L.LightningModule):
    def __init__(self, imgsize, N_samples, T_iterations, max_epochs, alpha, output_dim=128, lr=3e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = ResNet(in_channels=3, num_classes=output_dim)
        # buffer's current values can be loaded using the state_dict of the module which might be useful to know
        self.register_buffer("xi", torch.zeros(1,))  # weighted sum qij
        self.register_buffer("omega", torch.zeros(1,))  # count qij
        self.register_buffer("s_inv", torch.zeros(1,) + N_samples**2)  # scale parameter measure discrepancy between p and q
        self.register_buffer("alpha", torch.zeros(1,) + alpha)  # [0,1] adaptively extra attraction to ease training
        self.register_buffer("ro", torch.ones(1,))  # [0,1] forgetting rate s_inv
        self.register_buffer("N", torch.zeros(1,) + N_samples)  # N samples in dataset
        self.max_epochs = max_epochs
        self.T = T_iterations
        # KNN validation
        plaintraindataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=SwavEvalTransform(imgsize, num_views=1))
        plainvaldataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=SwavEvalTransform(imgsize, num_views=1))
        self.plain_trainloader = DataLoader(dataset=plaintraindataset, batch_size=512, shuffle=True, num_workers=20)
        self.plain_valloader = DataLoader(dataset=plainvaldataset, batch_size=512, shuffle=False, num_workers=20)

    def forward(self, x):
        return self.model(x)

    def on_train_epoch_start(self) -> None:
        # lr restarted by scheduler on epoch start
        self.xi = torch.zeros(1,).to(self.device)
        self.omega = torch.zeros(1,).to(self.device)
        self.eta = 1 - self.current_epoch / self.max_epochs

    def negative_forces(self):
        pass

    def positive_forces(self):
        pass

    def _shared_step(self, batch, batch_idx, mode="train"):
        # TODO how handle different batchsize? average, iterative compute?
        x_i, xhat_i, x_j = batch
        #import pdb
        #pdb.set_trace()
        B = x_i.shape[0]  # batchsize
        z = self.forward(torch.cat([x_i, xhat_i, x_j], dim=0))
        # TODO parametric or non-parametric similarity?
        z = z.norm(p=2, dim=1, keepdim=True)  # if cosine similarity
        z_i, zhat_i, z_j = z[0:B], z[B:2*B], z[2*B:3*B]
        # positive forces
        qii = torch.sum(z_i * zhat_i, dim=1) * 0.5 + 0.5  # cosine similarity [-1,1] -> [0,1]
        positive_forces = - torch.log(qii)
        self.xi = self.xi + self.alpha * qii.detach()
        self.omega = self.omega + self.alpha
        # negative forces
        qij = torch.sum(z_i * z_j, dim=1) * 0.5 + 0.5  # cosine similarity [-1,1] -> [0,1]
        negative_forces = qij / self.s_inv
        self.xi = self.xi + (1 - self.alpha) * qij.detach()
        self.omega = self.omega + (1 - self.alpha)

        # Update
        self.ro = self.N ** 2 / (self.N ** 2 + self.omega)
        self.s_inv = self.ro * self.s_inv + (1 - self.ro) * self.N**2 * self.xi / self.omega

        loss = positive_forces + negative_forces

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

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.lr,
        )
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html
        # start with initial lr, cosine anneal to 0 then restart.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0 = self.T,  # number of iterations for the first restart
        )

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


class SCLDataset(torch.utils.data.Dataset):
    def __init__(self, basedataset, transform):
        self.basedataset = basedataset
        self.transform = transform

    def __getitem__(self, item):
        # TODO improve sampling and clean code sample [1,N]^2
        x, _ = self.basedataset[item]
        x_i = self.transform(x)
        xhat_i = self.transform(x)
        j = torch.randint(low=0, high=len(self.basedataset), size=(1,))
        x_j, _ = self.basedataset[j]
        x_j = self.transform(x_j)
        return x_i, xhat_i, x_j

    def __len__(self):
        return len(self.basedataset)


if __name__ == "__main__":
    x = torch.rand(64, 3, 32, 32)
    model = ResNet(in_channels=3, num_classes=10)
    out = model(x)
    print(out.shape)

    # DATA AUGMENTATION
    COLOR_JITTER_STRENGTH = 0.5
    GAUSSIAN_BLUR = False
    IMG_SIZE = 32

    # PARAMS
    ALPHA = 0.5
    T_iterations = 1000

    # HYPERPARAMS
    BATCH_SIZE = 1 # currently only support batchsize 1
    NUM_WORKERS = 20
    MAX_EPOCHS = 100
    OPTIMIZER_NAME = "sgd"  #
    LR = 3e-4  # lr?

    traindataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        #transform=SwavTrainTransform(imgsize=IMG_SIZE, s=COLOR_JITTER_STRENGTH, gaus_blur=GAUSSIAN_BLUR),
        download=True
    )
    traindataset = SCLDataset(traindataset, transform=SwavTrainTransform(imgsize=IMG_SIZE, s=COLOR_JITTER_STRENGTH, gaus_blur=GAUSSIAN_BLUR, num_views=1))

    valdataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        #transform=SwavTrainTransform(imgsize=IMG_SIZE, s=COLOR_JITTER_STRENGTH, gaus_blur=GAUSSIAN_BLUR),
        download=True
    )
    valdataset = SCLDataset(valdataset, transform=SwavTrainTransform(imgsize=IMG_SIZE, s=COLOR_JITTER_STRENGTH, gaus_blur=GAUSSIAN_BLUR, num_views=1))

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

    torch.set_float32_matmul_precision('medium')

    out = next(iter(trainloader))
    #import pdb
    #pdb.set_trace()

    model = SCL(
        max_epochs=MAX_EPOCHS,
        imgsize=IMG_SIZE,
        lr=LR,
        alpha=ALPHA,
        N_samples=len(traindataset),
        T_iterations=T_iterations
    )

    logger = TensorBoardLogger("tb_logs", name="scl-cifar10")
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

