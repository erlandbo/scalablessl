from torch import nn
import torch
import lightning as L
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from sklearn.neighbors import KNeighborsClassifier
import math
import numpy as np
from ResNet import ResNet
from utils import load_dataset, load_knndataset


class SwAV(L.LightningModule):
    def __init__(self, imgsize, output_dim=128, num_prototypes=100, temp=0.1, lr=3e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = ResNet(in_channels=3, num_classes=output_dim)
        self.prototypes = nn.Linear(in_features=output_dim, out_features=num_prototypes)

        plaintraindataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=SwavEvalTransform(imgsize, num_views=1))
        plainvaldataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=SwavEvalTransform(imgsize, num_views=1))
        #train_dataset, val_dataset = torch.utils.data.random_split(plaindataset, [0.8, 0.2])
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
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=1e-6
        )
        return [optimizer] #, [scheduler]

    def on_validation_epoch_end(self):
        with torch.no_grad():
            knn = KNeighborsClassifier()
            X_train, X_val, y_train, y_val = [], [], [], []
            for X, y in self.plain_trainloader:
                #import pdb
                #pdb.set_trace()
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


def linear_warmup_cosine_anneal(warmup_steps, total_steps):
    # https://github.com/Lightning-Universe/lightning-bolts/blob/ba6b4c679ed72901923089ae01f5d4565aa7d12e/src/pl_bolts/optimizers/lr_scheduler.py#L128
    # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html
    def foo(step):
        if step < warmup_steps:
            eta = step / warmup_steps
        else:
            T_i = step - warmup_steps
            T_max = total_steps - warmup_steps
            eta = max(0.5 * (1 + math.cos(T_i / T_max * math.pi)), 0.0)
        return eta
    return foo


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
    TEMP = 0.1
    NUM_PROTOTYPES = 3000

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

    model = SwAV(
        imgsize=IMG_SIZE,
        temp=TEMP,
        num_prototypes=NUM_PROTOTYPES,
        lr=LR
    )

    #debug(trainloader, model)


    logger = TensorBoardLogger("tb_logs", name="swav-cifar10")
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

