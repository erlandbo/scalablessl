import torch
import lightning as L
from Datasets import SCLDataset, SCLFinetuneDataset
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from scl import SCL
from ImageAugmentations import SCLTrainTransform, SCLEvalTransform
from utils import load_imagedataset
from torch.utils.data import DataLoader, RandomSampler
import argparse

parser = argparse.ArgumentParser(description="SCL")

# Dataset and augmentation
parser.add_argument("--dataset", default="cifar10", type=str, help="Dataset to use")
parser.add_argument("--jitterstrength", default=0.5, type=float, help="COLOR_JITTER_STRENGTH")
parser.add_argument("--gausblur", default=False, type=bool, help="Dataset GAUSSIAN_BLUR")
parser.add_argument("--imgsize", default=32, type=int, help="IMG_SIZE")
parser.add_argument("--pflip", default=True, type=bool, help="PFLIP")

# Training hyperparameters
parser.add_argument("--batchsize", default=64, type=int, help="BATCH_SIZE")
parser.add_argument("--numworkers", default=0, type=int, help="NUM_WORKERS")
parser.add_argument("--lr", default=3e-4, type=float, help="LR_INIT")
parser.add_argument("--scheduler", default="None", type=str, help="scheduler: None, cosanneal, linwarmup_cosanneal")
parser.add_argument("--optimizer", default="adam", type=str, help="OPTIMIZER")
parser.add_argument("--valsplit", default=0.05, type=float, help="VAL_SPLIT")
parser.add_argument("--maxepochs", default=-1, type=int, help="Training will run t-iterations so epochs are disabled")

# SCL-algorithm hyperparameters
parser.add_argument("--titer", default=1_000_000, type=int, help="Number of iterations of training")
parser.add_argument("--alpha", default=0.5, type=float, help="ALPHA")
parser.add_argument("--ro", default=1.0, type=float, help="RO_INIT")
parser.add_argument("--xi", default=0.0, type=float, help="XI_INIT")
parser.add_argument("--omega", default=0.0, type=float, help="OMEGA_INIT")
parser.add_argument("--ncoeff", default=0.7, type=float, help="N_COEFF")
parser.add_argument("--sinv_init_coeff", default=2.0, type=float, help="INIT SINV = N_SAMPLES ** 2 / 10 ** sinv_init")
parser.add_argument("--simmetric", default="gaussian", type=str, help="SIMMETRIC: cossim, gaussian, stud-tkernel")
parser.add_argument("--sigma", default=2.0, type=float, help="std for gaussian-kernel")


# Model architecture
parser.add_argument("--modelarch", default="resnet18", type=str, help=
    "resnet9, resnet18, resnet34, resnet18torch, resnet34torch, resnet50torch, vit, vittorch, feedforward"
                    )

# ResNet
parser.add_argument("--in_channels", default=3, type=int)
parser.add_argument("--embed_dim", default=128, type=int, help="EMBED_DIM")
parser.add_argument("--normlayer", default=True, type=bool, help="use batchnorm in resnet")
parser.add_argument("--maxpool1", default=True, type=bool, help="use maxpool in first conv-layer")

# ViT
parser.add_argument("--transformer_patchdim", default=4, type=int)
parser.add_argument("--transformer_numlayers", default=6, type=int)
parser.add_argument("--transformer_dmodel", default=512, type=int)
parser.add_argument("--transformer_nhead", default=8, type=int)
parser.add_argument("--transformer_dff_ration", default=4, type=int)
parser.add_argument("--transformer_dropout", default=0.1, type=float)
parser.add_argument("--transformer_activation", default="relu", type=str)

# Finetune
parser.add_argument("--finetune_lr", default=3e-4, type=float)
parser.add_argument("--finetune_batchsize", default=64, type=int)
parser.add_argument("--finetune_knn", default=True, type=bool)
parser.add_argument("--finetune_linear", default=False, type=bool)
parser.add_argument("--finetune_interval", default=5, type=int)


def main():
    arg = parser.parse_args()

    traindataset, valdataset, testdataset, mean, std, num_classes = load_imagedataset(datasetname=arg.dataset, val_split=arg.valsplit)

    arg.nsamples = len(traindataset)
    arg.numclasses = num_classes

    # Train and val dataloader
    # NOTE: sampling with replacement
    trainloader = DataLoader(
        SCLDataset(
            traindataset,
            transform=SCLTrainTransform(
                imgsize=arg.imgsize,
                mean=mean,
                std=std,
                s=arg.jitterstrength,
                gaus_blur=arg.gausblur,
                num_views=1,
                p_flip=arg.pflip
            ),
        ),
        batch_size=arg.batchsize,
        num_workers=arg.numworkers,
        sampler=RandomSampler(traindataset, replacement=True)  # Sample randomly with replacement
    )
    valloader = DataLoader(
        SCLDataset(
            valdataset,
            transform=SCLTrainTransform(
                imgsize=arg.imgsize,
                mean=mean,
                std=std,
                s=arg.jitterstrength,
                gaus_blur=arg.gausblur,
                num_views=1,
                p_flip=arg.pflip
            ),
        ),
        batch_size=arg.batchsize,
        num_workers=arg.numworkers,
        sampler=RandomSampler(valdataset, replacement=True)  # Sample randomly with replacement
    )

    # Finetune dataloaders
    finetune_traindataset = SCLFinetuneDataset(
        traindataset,
        transform=SCLEvalTransform(imgsize=arg.imgsize, mean=mean, std=std, num_views=1),
    )
    finetune_testloader = SCLFinetuneDataset(
        testdataset,
        transform=SCLEvalTransform(imgsize=arg.imgsize, mean=mean, std=std, num_views=1),
    )

    model = SCL(arg)

    # TODO better ways to set finetune-dataset?
    model.load_finetune_dataset(finetune_traindataset, finetune_testloader)

    # Lightning
    torch.set_float32_matmul_precision('medium')

    logger = TensorBoardLogger("tb_logs", name="scl")
    trainer = L.Trainer(
        logger=logger,
        max_epochs=arg.maxepochs,
        max_steps=arg.titer,
        precision=32,
        accelerator="gpu",
        callbacks=[
            ModelCheckpoint(
                save_weights_only=True,
                mode="max",
                monitor="knn_test_acc",
                save_last=True,
                save_on_train_epoch_end=False  # save on val-epoch end
            ),
            LearningRateMonitor("step")
        ]
    )

    trainer.fit(model, train_dataloaders=trainloader, val_dataloaders=valloader)

    # Post training
    #BASEMODEL_CHECKPOINT = "tb_logs/scl/version_6/checkpoints/epoch=271-step=53312.ckpt"
    #model = model.load_from_checkpoint(BASEMODEL_CHECKPOINT)

    # compute_embeddings(model, plain_valloader, logpath="./tb_logs/scl/embeddings0/")


if __name__ == "__main__":
    main()
    test = 10