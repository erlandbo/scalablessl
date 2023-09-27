import torch
import lightning as L
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from ResNet import ResNet


class BHCL(L.LightningModule):
    def __init__(self, imgsize, output_dim=128, num_clusters=100, temp=0.1, lr=3e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = ResNet(in_channels=3, num_classes=output_dim)
        self.register_buffer("clusters", torch.rand(num_clusters, output_dim))
        self.num_clusters = num_clusters

        plaintraindataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=SwavEvalTransform(imgsize, num_views=1))
        plainvaldataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=SwavEvalTransform(imgsize, num_views=1))
        self.plain_trainloader = DataLoader(dataset=plaintraindataset, batch_size=512, shuffle=True, num_workers=20)
        self.plain_valloader = DataLoader(dataset=plainvaldataset, batch_size=512, shuffle=False, num_workers=20)

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx, mode="train"):
        images, y = batch
        xt, xs = images
        z = self.forward(torch.cat([xt, xs], dim=0))
        z = F.normalize(z, p=2, dim=1)
        distances = torch.cdist(z, self.clusters[None,...]).squeeze()  # (B,E) (1,C,E) -> (1,B,C) -> (B,C)
        c = torch.argmin(distances, dim=1, keepdim=True)  # ()
        for i in range(self.num_clusters):
            ci = torch.mean() z[c == i]
        #c_updates = torch.mean(distances[ci])
        B = z.shape[0] // 2
        z_t = z[:B]
        z_s = z[B:]
        p_t = F.softmax(z_t / self.hparams.temp, dim=1)
        p_s = F.softmax(z_s / self.hparams.temp, dim=1)
        loss = torch.mean(-0.5 * torch.sum(p_t * torch.log(p_s), dim=1) + -0.5 * torch.sum(p_s * torch.log(p_t), dim=1))
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
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
        )
        return [optimizer]  #, [scheduler]

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


class SwavTrainTransform():
    # augmentations as described in SimCLR paper
    def __init__(self, imgsize=32, s=0.5, gaus_blur=False, num_views=2):
        self.num_views = num_views
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        transform = [
            transforms.RandomResizedCrop(imgsize, scale=(0.14, 1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=.2)
        ]
        if gaus_blur:
            transform.append(transforms.GaussianBlur(kernel_size=int(imgsize*0.1), sigma=(0.1, 2.0)))
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]))  # CiFar10
        self.transform = transforms.Compose(transform)

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.num_views)]


class SwavEvalTransform():
    def __init__(self, imgsize=32, num_views=2):
        self.num_views = num_views
        self.transform = transforms.Compose([
            transforms.Resize(imgsize),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])  # CiFar10
        ])

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.num_views)] if self.num_views > 1 else self.transform(x)


if __name__ == "__main__":
    # DATA AUGMENTATION
    COLOR_JITTER_STRENGTH = 0.5
    GAUSSIAN_BLUR = False
    IMG_SIZE = 32

    # PARAMS
    TEMP = 0.1
    NUM_CLUSTERS = 3000

    # HYPERPARAMS
    BATCH_SIZE = 256
    NUM_WORKERS = 20
    MAX_EPOCHS = 50
    OPTIMIZER_NAME = "adam"  # "LARS"
    LR = 3e-4  # 0.075 * BATCH_SIZE ** 0.5
    WEIGHT_DECAY = 1e-6

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

    trainloader = DataLoader(traindataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valloader = DataLoader(valdataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

    torch.set_float32_matmul_precision('medium')

    model = BHCL(
        imgsize=IMG_SIZE,
        temp=TEMP,
        num_clusters=NUM_CLUSTERS,
        lr=LR
    )

    logger = TensorBoardLogger("tb_logs", name="bhcl")
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

