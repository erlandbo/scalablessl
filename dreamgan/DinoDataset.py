import torchvision
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as T


class DinoAugment():
    def __init__(self, num_local_crops, imgsize=224):
        self.num_local_crops = num_local_crops
        self.global_transform1 = T.Compose([
            T.RandomResizedCrop(imgsize, scale=(0.4, 1), interpolation=Image.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=1.0),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.global_transform2 = T.Compose([
            T.RandomResizedCrop(imgsize, scale=(0.4, 1), interpolation=Image.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.1),
            # T.RandomSolarize( , p=0.2),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.local_transformation = T.Compose([
            T.RandomResizedCrop(imgsize, scale=(0.05, 0.4), interpolation=Image.BICUBIC),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.RandomApply([T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.5),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def __call__(self, x):
        crops = []
        crops.append(self.global_transform1(x))
        crops.append(self.global_transform2(x))
        for _ in range(self.num_local_crops):
            crops.append(self.local_transformation(x))
        return crops



def plain_transform(imgsize):
    return T.Compose([
        T.RandomResizedCrop(imgsize, scale=(0.4, 1), interpolation=Image.BICUBIC),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])