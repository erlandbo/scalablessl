from torchvision import transforms


class SimCLRTrainTransform():
    # augmentations as described in SimCLR paper
    def __init__(self, imgsize=32, s=0.5, gaus_blur=False, num_views=2, mean=(0.5,), std=(0.5,)):
        self.num_views = num_views
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        transform = [
            transforms.RandomResizedCrop(imgsize, scale=(0.14, 1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=.2)
        ]
        if gaus_blur:
            transform.append(transforms.GaussianBlur(kernel_size=int(imgsize*0.1), sigma=(0.1, 2.0)))
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(mean, std))
        self.transform = transforms.Compose(transform)

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.num_views)] if self.num_views > 1 else self.transform(x)


class SimCLREvalTransform():
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


class MNISTTrainTransform():
    def __init__(self, imgsize=32, s=0.5, gaus_blur=False, num_views=2):
        self.num_views = num_views
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        transform = [
            transforms.Resize(imgsize),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=.2)
        ]
        if gaus_blur:
            transform.append(transforms.GaussianBlur(kernel_size=int(imgsize*0.1), sigma=(0.1, 2.0)))
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize((0.1307,), (0.3081,)))  # mnist
        self.transform = transforms.Compose(transform)

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.num_views)] if self.num_views > 1 else self.transform(x)


class MNISTEvalTransform():
    def __init__(self, imgsize=32, num_views=2):
        self.num_views = num_views
        transform = [
            transforms.Resize(imgsize),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
        self.transform = transforms.Compose(transform)

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.num_views)] if self.num_views > 1 else self.transform(x)