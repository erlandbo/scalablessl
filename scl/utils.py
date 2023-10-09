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
        "stl10": [(0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)],
    }
    mu, sigma = (0.5,), (0.5,)
    if dataset in image_stats.keys():
        mu, sigma = image_stats[dataset]
    return mu, sigma


def load_imagedataset(datasetname, val_split=0.2):
    traindataset, testdataset = None, None
    assert datasetname in ["cifar10", "mnist", "fashionmnist", "svhn", "celeba", "stl10"]
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
        num_classes = 10
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
        num_classes = 10
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
    elif datasetname == "celeba":
        traindataset = torchvision.datasets.CelebA(
            root="./data",
            split="train",
            download=True,
            transform=None
        )
        testdataset = torchvision.datasets.CelebA(
            root="./data",
            split="test",
            download=True,
            transform=None
        )
        num_classes = 10
    elif datasetname == "stl10":
        traindataset = torchvision.datasets.STL10(
            root="./data",
            split="train+unlabeled",
            download=True,
            transform=None
        )
        testdataset = torchvision.datasets.STL10(
            root="./data",
            split="test",
            download=True,
            transform=None
        )
        num_classes = 10

    mean, std = get_image_stats(datasetname)

    trainsize = int((1.0 - val_split) * len(traindataset))
    valsize = len(traindataset) - trainsize
    traindataset, valdataset = random_split(traindataset, lengths=[trainsize, valsize])
    return traindataset, valdataset, testdataset, mean, std, num_classes

