import torch
import lightning as L
from Datasets import SCLDataset, SCLFinetuneDataset
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from simclr import SimCLR
from ImageAugmentations import SCLTrainTransform, SCLEvalTransform, SCLMnistTrainTransform
from utils import load_imagedataset
from torch.utils.data import DataLoader, RandomSampler
import argparse

parser = argparse.ArgumentParser(description="SimCLR")

# Dataset and augmentation
parser.add_argument("--dataset", default="cifar10", type=str, help="Dataset to use")
parser.add_argument("--jitterstrength", default=0.5, type=float)
parser.add_argument("--gausblur", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--imgsize", default=32, type=int)
parser.add_argument("--pflip", default=0.5, type=float, help="prob random horizontal-plip")

parser.add_argument("--aug", default="simclr", type=str, help="Augmentation to use: simclr, mnist")

# Training hyperparameters
parser.add_argument("--batchsize", default=64, type=int)
parser.add_argument("--numworkers", default=0, type=int)

parser.add_argument("--lr", default=3e-4, type=float)
parser.add_argument("--min_lr", default=0.0, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--lr_warmup_steps", default=10, type=int)
parser.add_argument("--lr_total_steps", default=100, type=int)
parser.add_argument("--scheduler", default="None", type=str ) # "scheduler: None, cosanneal, linwarmup_cosanneal"
parser.add_argument("--lr_scheduler_interval", default="epoch", type=str )  # "epoch, step"
parser.add_argument("--optimizer", default="adam", type=str)
parser.add_argument("--valsplit", default=0.05, type=float)
parser.add_argument("--maxepochs", default=-1, type=int)
parser.add_argument("--maxtime", default="", type=str)  # default None else format DD:HH:MM:SS (days, hours, minutes seconds)

# SCL-algorithm hyperparameters
parser.add_argument("--titer", default=-1, type=int, help="Number of iterations of training")
parser.add_argument("--simmetric", default="euclidean", type=str, help="SIMMETRIC: cosine, euclidean, gauss")

# Model architecture
# resnet9, resnet18, resnet34, resnet18torch, resnet34torch, resnet50torch, vit, vittorch, feedforward
parser.add_argument("--modelarch", default="resnet18", type=str)
parser.add_argument("--embed_dim", default=128, type=int)
parser.add_argument("--machine_epsilon", default=1e-6, type=float, help="clamp min eps values L2-distance")

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

# Finetune
parser.add_argument("--finetune_lr", default=3e-4, type=float)
parser.add_argument("--finetune_batchsize", default=64, type=int)
parser.add_argument("--finetune_knn", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--finetune_n_neighbours", default=20, type=int)  # swav-paper uses nn=20 and nn=200
parser.add_argument("--finetune_linear", default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--finetune_interval", default=10, type=int)

# Plotting
parser.add_argument("--plot2d_interval", default=0, type=int)

parser.add_argument("--experiment_name", default="experiment", type=str)


def main():
    arg = parser.parse_args()

    traindataset, valdataset, testdataset, mean, std, num_classes, knn_traindataset, knn_testdataset = load_imagedataset(datasetname=arg.dataset, val_split=arg.valsplit)

    arg.nsamples = len(traindataset)
    arg.numclasses = num_classes

    # Train and val dataloader
    if arg.aug == "simclr":
        train_transform = SCLTrainTransform(
            imgsize=arg.imgsize,
            mean=mean,
            std=std,
            s=arg.jitterstrength,
            gaus_blur=arg.gausblur,
            num_views=1,
            p_flip=arg.pflip
        )
        val_transform = SCLTrainTransform(
            imgsize=arg.imgsize,
            mean=mean,
            std=std,
            s=arg.jitterstrength,
            gaus_blur=arg.gausblur,
            num_views=1,
            p_flip=arg.pflip
        )
    else:
        train_transform = SCLMnistTrainTransform(
            imgsize=arg.imgsize,
            mean=mean,
            std=std,
            gaus_blur=arg.gausblur,
            num_views=1,
            p_flip=arg.pflip
        )
        val_transform = SCLMnistTrainTransform(
            imgsize=arg.imgsize,
            mean=mean,
            std=std,
            gaus_blur=arg.gausblur,
            num_views=1,
            p_flip=arg.pflip
        )

    # NOTE: sampling with replacement

    trainloader = DataLoader(
        SCLDataset(
            traindataset,
            transform=train_transform
        ),
        batch_size=arg.batchsize,
        num_workers=arg.numworkers,
        sampler=RandomSampler(traindataset, replacement=True)  # Sample randomly with replacement
    )
    valloader = DataLoader(
        SCLDataset(
            valdataset,
            transform=val_transform
        ),
        batch_size=arg.batchsize,
        num_workers=arg.numworkers,
        sampler=RandomSampler(valdataset, replacement=True)  # Sample randomly with replacement
    )

    # Finetune dataloaders
    finetune_traindataset = SCLFinetuneDataset(
        knn_traindataset,
        transform=SCLEvalTransform(imgsize=arg.imgsize, mean=mean, std=std, num_views=1),
    )
    finetune_testdataset = SCLFinetuneDataset(
        knn_testdataset,
        transform=SCLEvalTransform(imgsize=arg.imgsize, mean=mean, std=std, num_views=1),
    )

    model = SimCLR(arg)

    # TODO better ways to set finetune-dataset?
    model.load_finetune_dataset(finetune_traindataset, finetune_testdataset)

    # Lightning
    torch.set_float32_matmul_precision('medium')
    run_name = "/{}_{}_batch_{}_Ncoeff_{}_embeddim_{}_alpha_{}".format(arg.modelarch, arg.dataset, arg.batchsize, arg.ncoeff, arg.embed_dim, arg.alpha)
    logger = TensorBoardLogger("tb_logs", name="scl/"+ arg.experiment_name + run_name)
    trainer = L.Trainer(
        logger=logger,
        max_epochs=arg.maxepochs,
        max_time = arg.maxtime if arg.maxtime != "" else None,
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