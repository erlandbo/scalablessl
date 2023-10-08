import torchvision
from ImageAugmentations import SCLEvalTransform
from torch.utils.data import random_split


def get_image_stats(dataset):
    image_stats = {
        "cifar10": [(0.4915, 0.4823, 0.4468), (0.2470, 0.2435, 0.2616)],
        "mnist": [(0.5,), (0.5,)],
        "fashionmnist": [(0.5,), (0.5,)],
        "svhn": [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)],
        "celeba": [(0.5, 0.5, 0.5), (0.5, 0.5, 0.5)],
    }
    mu, sigma = (0.5,), (0.5,)
    if dataset in image_stats.keys():
        mu, sigma = image_stats[dataset]
    return mu, sigma


def load_imagedataset(datasetname, val_split=0.2):
    traindataset, testdataset = None, None
    mean, std = (0.5,), (0.5,)
    num_classes = 0
    if datasetname == "cifar10":
        traindataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=None
        )
        testdataset = torchvision.datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=None
        )
        mean, std = get_image_stats(datasetname)
        num_classes = 10
    elif datasetname == "mnist":
        traindataset = torchvision.datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=None
        )
        testdataset = torchvision.datasets.MNIST(
            root="./data",
            train=True,
            download=True,
            transform=None
        )
    elif datasetname == "fashionmnist":
        traindataset = torchvision.datasets.FashionMNIST(
            root="./data",
            train=True,
            download=True,
            transform=None
        )
        testdataset = torchvision.datasets.FashionMNIST(
            root="./data",
            train=True,
            download=True,
            transform=None
        )
    elif datasetname == "svhn":
        traindataset = torchvision.datasets.SVHN(
            root="./data",
            split="train",
            download=True,
            transform=None
        )
        testdataset = torchvision.datasets.SVHN(
            root="./data",
            split="train",
            download=True,
            transform=None
        )
    elif datasetname == "svhn":
        traindataset = torchvision.datasets.CelebA(
            root="./data",
            split="train",
            download=True,
            transform=None
        )
        testdataset = torchvision.datasets.CelebA(
            root="./data",
            split="train",
            download=True,
            transform=None
        )
    trainsize = int((1.0 - val_split) * len(traindataset))
    valsize = len(traindataset) - trainsize
    traindataset, valdataset = random_split(traindataset, lengths=[trainsize, valsize])
    return traindataset, valdataset, testdataset, mean, std, num_classes

