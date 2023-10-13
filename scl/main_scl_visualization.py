from Datasets import SCLFinetuneDataset
from scl_visualization import SCLVisualizer
from scl import SCL
from ImageAugmentations import SCLEvalTransform
from utils import load_imagedataset
from torch.utils.data import DataLoader
import argparse
import torch

parser = argparse.ArgumentParser(description="SCL visualization")

# Dataset
parser.add_argument("--dataset", default="cifar10", type=str, help="Dataset to use")
parser.add_argument("--imgsize", default=32, type=int)

# Training hyperparameters
parser.add_argument("--batchsize", default=64, type=int)
parser.add_argument("--numworkers", default=0, type=int)
parser.add_argument("--valsplit", default=0.1, type=float)

parser.add_argument("--checkpoint_path", type=str)

parser.add_argument("--plot_name", type=str)


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

    sclmodel = SCL.load_from_checkpoint(arg.checkpoint_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    scl_ploter = SCLVisualizer(device=device, sclmodel=sclmodel)

    scl_ploter.fit(trainloader, testloader)

    scl_ploter.tensorboard_projector(testloader, mu=mean, sigma=std, tag=arg.dataset + "_test")


if __name__ == "__main__":
    main()
