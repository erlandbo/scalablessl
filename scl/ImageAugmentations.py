from torchvision import transforms


class SCLTrainTransform():
    # augmentations as described in SimCLR paper
    def __init__(self, imgsize, mean, std, s=0.5, gaus_blur=False, num_views=2, p_flip=0.5):
        self.num_views = num_views
        color_jitter = transforms.ColorJitter(
            brightness=0.8*s,
            contrast=0.8*s,
            saturation=0.8*s,
            hue=0.2*s
        )
        transform = [
            transforms.RandomResizedCrop(size=imgsize),  #, scale=(0.14, 1)),
            transforms.RandomHorizontalFlip(p=p_flip),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ]
        if gaus_blur:
            transform.append(transforms.GaussianBlur(kernel_size=int(imgsize*0.1), sigma=(0.1, 2.0)))
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean, std))
        self.transform = transforms.Compose(transform)

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.num_views)] if self.num_views > 1 else self.transform(x)


class SCLEvalTransform():
    def __init__(self, imgsize=32, num_views=2, mean=(0.5), std=(0.5,)):
        self.num_views = num_views
        transform = [
            transforms.Resize(imgsize),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
        self.transform = transforms.Compose(transform)

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.num_views)] if self.num_views > 1 else self.transform(x)

