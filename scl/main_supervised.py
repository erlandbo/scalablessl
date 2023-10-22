import torch
import lightning as L
from Datasets import SCLFinetuneDataset
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from supervised import Supervised
from ImageAugmentations import SCLEvalTransform
from utils import load_imagedataset
from torch.utils.data import DataLoader
import argparse
from scl import SCL

parser = argparse.ArgumentParser(description="Supervised image classification")

# Dataset and augmentation
parser.add_argument("--dataset", default="cifar10", type=str, help="Dataset to use")
parser.add_argument("--imgsize", default=32, type=int)

# Model architecture
# resnet9, resnet18, resnet34, resnet18torch, resnet34torch, resnet50torch, vit, vittorch, feedforward
parser.add_argument("--modelarch", default="resnet18", type=str)
parser.add_argument("--embed_dim", default=128, type=int)


# ResNet
parser.add_argument("--in_channels", default=3, type=int)
parser.add_argument("--normlayer", default=True, action=argparse.BooleanOptionalAction, help="use batchnorm in resnet")
parser.add_argument("--maxpool1", default=False, action=argparse.BooleanOptionalAction, help="use maxpool in first conv-layer")
parser.add_argument("--first_conv", default=False, action=argparse.BooleanOptionalAction, help="use maxpool in first conv-layer")

# ViT
parser.add_argument("--transformer_patchdim", default=4, type=int)
parser.add_argument("--transformer_numlayers", default=6, type=int)
parser.add_argument("--transformer_dmodel", default=512, type=int)
parser.add_argument("--transformer_nhead", default=8, type=int)
parser.add_argument("--transformer_dff_ration", default=4, type=int)
parser.add_argument("--transformer_dropout", default=0.1, type=float)
parser.add_argument("--transformer_activation", default="relu", type=str)


# Training hyperparameters
parser.add_argument("--batchsize", default=64, type=int)
parser.add_argument("--numworkers", default=0, type=int)
parser.add_argument("--lr", default=3e-4, type=float)
parser.add_argument("--optimizer", default="adam", type=str)
parser.add_argument("--valsplit", default=0.1, type=float)
parser.add_argument("--maxepochs", default=100, type=int)
parser.add_argument("--scheduler", default="None", type=str ) # "scheduler: None, cosanneal, linwarmup_cosanneal"

# Model architecture from SCL
parser.add_argument("--num_classes", default=10, type=float)

parser.add_argument("--checkpoint_path", default="", type=str)


def main():
    arg = parser.parse_args()

    traindataset, valdataset, testdataset, mean, std, num_classes, knn_traindataset, knn_testdataset = load_imagedataset(datasetname=arg.dataset, val_split=arg.valsplit)

    arg.numclasses = num_classes

    trainloader = DataLoader(
        SCLFinetuneDataset(
            knn_traindataset,
            transform=SCLEvalTransform(imgsize=arg.imgsize, mean=mean, std=std, num_views=1)
        ),
        batch_size=arg.batchsize,
        num_workers=arg.numworkers,
    )
    valloader = DataLoader(
        SCLFinetuneDataset(
            knn_testdataset,
            transform=SCLEvalTransform(imgsize=arg.imgsize, mean=mean, std=std, num_views=1)
        ),
        batch_size=arg.batchsize,
        num_workers=arg.numworkers,
    )

    if arg.checkpoint_path:
        pretrained_model = SCL.load_from_checkpoint(arg.checkpoint_path)
    else:
        pretrained_model = None
    model = Supervised(arg, pretrained_model)

    # Lightning
    torch.set_float32_matmul_precision('medium')

    logger = TensorBoardLogger("tb_logs", name="supervised")
    trainer = L.Trainer(
        logger=logger,
        max_epochs=arg.maxepochs,
        precision=32,
        accelerator="gpu",
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True,
                mode="min",
                monitor="val_loss",
                save_last=True,
            ),
        ]
    )

    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)


if __name__ == "__main__":
    main()
    test = 10