import torchvision
from Augmentations import SwavTrainTransform, SwavEvalTransform


def load_dataset(name):
    traindataset, valdataset = None, None
    if name == "cifar10":
        traindataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=True,
            download=True
        )
        valdataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=False,
            download=True
        )
    elif name == "mnist":
        traindataset = torchvision.datasets.MNIST(
            root="./data",
            train=True,
            download=True
        )
        valdataset = torchvision.datasets.MNIST(
            root="./data",
            train=False,
            download=True
        )
    return traindataset, valdataset


def load_knndataset(name, imgsize):
    plaintraindataset, plainvaldataset = None, None
    if name == "cifar10":
        plaintraindataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=SwavEvalTransform(imgsize, num_views=1, dataset=name)
        )
        plainvaldataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=SwavEvalTransform(imgsize, num_views=1, dataset=name)
        )
    elif name == "mnist":
        plaintraindataset = torchvision.datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=SwavEvalTransform(imgsize, num_views=1, dataset=name)
        )
        plainvaldataset = torchvision.datasets.MNIST(
            root="./data",
            train=False,
            download=True,
            transform=SwavEvalTransform(imgsize, num_views=1, dataset=name)
        )

    return plaintraindataset, plainvaldataset