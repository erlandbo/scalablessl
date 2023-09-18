import torch
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.models.resnet import resnet18
import numpy as np
import torch.func as fc
from functools import partial
from torch.nn import functional as F

"""
Implementation of - A Simple Framework for Contrastive Learning of Visual Representations, Chen et.al.,2020
Forward AD: https://github.com/orobix/fwdgrad/tree/main
"""


class SimCLRTrainTransform():
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


class SimCLREvalTransform():
    def __init__(self, imgsize, num_views=2):
        self.num_views = num_views
        self.transform = transforms.Compose([
            transforms.Resize(imgsize),
            transforms.ToTensor(),
            transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])  # CiFar10
        ])

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.num_views)]


class SimCLRModel(nn.Module):
    def __init__(self,
                 h_dim=128*4,
                 z_dim=128,
                 ):
        super().__init__()
        convnet = []
        resmodel = resnet18(norm_layer=partial(nn.BatchNorm2d, track_running_stats=False))
        for name, layer in resmodel.named_children():
            if isinstance(layer, nn.Linear) or isinstance(layer, nn.MaxPool2d):
                continue
            if name == "conv1":
                layer = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
            convnet.append(layer)

        self.f = nn.Sequential(*convnet)
        self.g = nn.Sequential(
            nn.Linear(512, h_dim, bias=False),
            #nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim, bias=True)
        )

    def forward(self, x):
        h = torch.flatten(self.f(x), start_dim=1)
        z = self.g(h)
        return h, z


def ad_func(params, buffers, names, model, x, y):
    #h, z = model(x)
    #loss = F.cross_entropy(z, y)
    # x = torch.cat([xi, xj], dim=0)
    #h, z = model.forward(x)
    h, z = fc.functional_call(model, ({k: v for k, v in zip(names, params)}, buffers), (x,))
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


class Trainer():
    def __init__(self,
                 model,
                 lr,
                 temp,
                 device
                 ):
        self.device = device
        self.model: nn.Module = model
        self.model.to(device)
        self.temp = temp
        self.lr = lr
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

    def train(self, trainloader, valloader, max_epochs):
        for epoch in range(max_epochs):
            train_loss, train_top1, train_top5 = [], [], []
            self.model.train()
            named_buffers = dict(self.model.named_buffers())
            named_params = dict(self.model.named_parameters())
            names = named_params.keys()
            params = named_params.values()

            for (images, y) in trainloader:
                xi, xj = images
                xi, xj, y = xi.to(self.device), xj.to(self.device), y.to(self.device)
                x = torch.cat([xi, xj])
                # h, z = self.model.forward(torch.cat([xi, xj], dim=0))
                self.optimizer.zero_grad()
                v_params = tuple([torch.randn_like(param) for param in params])
                foo = partial(
                    ad_func,
                    model=self.model,
                    names=names,
                    buffers=named_buffers,
                    x=x,
                    y=y
                )
                loss, jvp = fc.jvp(foo, (tuple(params),), (v_params,))
                #loss, top1, top5, mean_pos = loss
                for v, p in zip(v_params, params):
                    p.grad = v * jvp
                # loss.backward()
                self.optimizer.step()
                train_loss.append(loss.item())
                #train_top1.append(top1.item())
                #train_top5.append(top5.item())
            print("Training")
            print("EPOCH: ", epoch)
            print("LOSS: ", np.mean(train_loss))
            #print("TOP1: ", np.mean(train_top1))
            #print("TOP5: ", np.mean(train_top5))
            print("--------------------------------")

            val_loss, val_top1, val_top5 = [], [], []
            self.model.eval()
            for (images, y) in valloader:
                with torch.no_grad():
                    xi, xj = images
                    xi, xj, y = xi.to(self.device), xj.to(self.device), y.to(self.device)
                    h, z = self.model.forward(torch.cat([xi, xj], dim=0))
                    loss, top1, top5, mean_pos = self.ntXentLoss(z)
                    val_loss.append(loss.item())
                    val_top1.append(top1.item())
                    val_top5.append(top5.item())
                    #import pdb
                    #pdb.set_trace()
            print("Validation")
            print("EPOCH: ", epoch)
            print("LOSS: ", np.mean(val_loss))
            print("TOP1: ", np.mean(val_top1))
            print("TOP5: ", np.mean(val_top5))
            print("--------------------------------")

    def ntXentLoss(self, z):
        # x = torch.cat([xi, xj], dim=0)
        # h, z = self.model.forward(x)
        # ((2N, g) @ (g, 2N)) / (2N,1) @ (1,2N) -> (2N, 2N) / (2N,2N)
        sim_matrix = (z @ z.T) / (z.norm(p=2, dim=1, keepdim=True) @ z.norm(p=2, dim=1, keepdim=True).T)
        mask = torch.eye(z.shape[0], dtype=torch.bool, device=z.device)
        pos_mask = mask.roll(shifts=sim_matrix.shape[0]//2, dims=1).bool()  # find pos-pair N away
        pos = torch.exp(sim_matrix[pos_mask] / self.temp)
        neg = torch.exp(sim_matrix.masked_fill(mask, value=float("-inf")) / self.temp)
        loss = -torch.log(pos / torch.sum(neg))
        #loss = - (sim_matrix[pos_mask] / self.hparams.temp / 2) + (torch.logsumexp(sim_matrix.masked_fill(mask, value=float("-inf")) / self.hparams.temp, dim=1) / 2)
        # Find the rank for the positive pair
        sim_matrix = torch.cat([sim_matrix[pos_mask].unsqueeze(1), sim_matrix.masked_fill(pos_mask,float("-inf"))], dim=1)
        pos_pair_pos = torch.argsort(sim_matrix, descending=True, dim=1).argmin(dim=1)
        top1 = torch.mean((pos_pair_pos == 0).float())
        top5 = torch.mean((pos_pair_pos < 5).float())
        mean_pos = torch.mean(pos_pair_pos.float())
        return torch.mean(loss), top1, top5, mean_pos


if __name__ == "__main__":
    # DATA AUGMENTATION
    COLOR_JITTER_STRENGTH = 0.5
    GAUSSIAN_BLUR = False
    IMG_SIZE = 32

    # PRE-TRAIN MODEl
    TEMPERATURE = 0.07
    Z_DIM = 128
    H_DIM = 128 * 4

    # HYPERPARAMS
    BATCH_SIZE = 256
    NUM_WORKERS = 20
    MAX_EPOCHS = 500
    OPTIMIZER_NAME = "sgd"  # "LARS"
    LR = 2e-4  # 0.075 * BATCH_SIZE ** 0.5
    WEIGHT_DECAY = 1e-4

    #torch.set_float32_matmul_precision('medium')  # | 'high')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SimCLRModel(h_dim=H_DIM, z_dim=Z_DIM)

    traindataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        transform=SimCLRTrainTransform(imgsize=IMG_SIZE, gaus_blur=GAUSSIAN_BLUR, s=COLOR_JITTER_STRENGTH),
        download=True
    )
    valdataset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        transform=SimCLRTrainTransform(imgsize=IMG_SIZE),
        download=True
    )

    traindataloader = DataLoader(traindataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    valdataloader = DataLoader(valdataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)

    trainer = Trainer(model=model, lr=LR, temp=TEMPERATURE, device=device)
    trainer.train(traindataloader, valdataloader, max_epochs=MAX_EPOCHS)

