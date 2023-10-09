import torch
import lightning as L
from Datasets import SCLFinetuneDataset
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from scl_offline_finetuner import SCLOfflineFinetuner
from scl import SCL
from ImageAugmentations import SCLEvalTransform
from utils import load_imagedataset
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser(description="SCL Offline finetuner")

# Dataset and augmentation
parser.add_argument("--dataset", default="cifar10", type=str, help="Dataset to use")
parser.add_argument("--imgsize", default=32, type=int)

# Training hyperparameters
parser.add_argument("--batchsize", default=64, type=int)
parser.add_argument("--numworkers", default=0, type=int)
parser.add_argument("--lr", default=3e-4, type=float)
parser.add_argument("--optimizer", default="adam", type=str)
parser.add_argument("--valsplit", default=0.1, type=float)
parser.add_argument("--maxepochs", default=100, type=int)

# Model architecture from SCL
parser.add_argument("--hdim", default=512, type=int)
parser.add_argument("--num_classes", default=10, type=float)
parser.add_argument("--activation", default="relu", type=str)

parser.add_argument("--checkpoint_path", type=str)


def main():
    arg = parser.parse_args()

    traindataset, valdataset, testdataset, mean, std, num_classes = load_imagedataset(datasetname=arg.dataset, val_split=arg.valsplit)

    arg.numclasses = num_classes

    trainloader = DataLoader(
        SCLFinetuneDataset(
            traindataset,
            transform=SCLEvalTransform(imgsize=arg.imgsize, mean=mean, std=std, num_views=1)
        ),
        batch_size=arg.batchsize,
        num_workers=arg.numworkers,
    )
    valloader = DataLoader(
        SCLFinetuneDataset(
            valdataset,
            transform=SCLEvalTransform(imgsize=arg.imgsize, mean=mean, std=std, num_views=1)
        ),
        batch_size=arg.batchsize,
        num_workers=arg.numworkers,
    )

    testloader = DataLoader(
        SCLFinetuneDataset(
            testdataset,
            transform=SCLEvalTransform(imgsize=arg.imgsize, mean=mean, std=std, num_views=1)
        ),
        batch_size=arg.batchsize,
        num_workers=arg.numworkers,
    )

    backbone = SCL.load_from_checkpoint(arg.checkpoint_path)
    finetuner = SCLOfflineFinetuner(backbone_module=backbone, hparams=arg)

    # Lightning
    torch.set_float32_matmul_precision('medium')

    logger = TensorBoardLogger("tb_logs", name="scl-offline-finetuner")
    trainer = L.Trainer(
        logger=logger,
        max_epochs=arg.maxepochs,
        max_steps=arg.titer,
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

    trainer.fit(finetuner, train_dataloaders=trainloader, val_dataloaders=valloader)

    trainer.test(model=finetuner, ckpt_path="best", dataloaders=testloader, verbose=True)


if __name__ == "__main__":
    main()
    test = 10