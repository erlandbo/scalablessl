from torchvision import transforms


class SimCLRTrainTransform():
    # augmentations as described in SimCLR paper
    def __init__(self, imgsize=32, s=0.4, gaus_blur=False, num_views=2):
        self.num_views = num_views
        # color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        # transform = [
        #     transforms.RandomResizedCrop(imgsize, scale=(0.14, 1)),
        #     transforms.RandomHorizontalFlip(p=0.5),
        #     transforms.RandomApply([color_jitter], p=0.8),
        #     transforms.RandomGrayscale(p=.2)
        # ]
        # if gaus_blur:
        #     transform.append(transforms.GaussianBlur(kernel_size=int(imgsize*0.1), sigma=(0.1, 2.0)))
        # transform.append(transforms.ToTensor())
        # transform.append(transforms.Normalize(0.5, 0.5))  # CiFar10
        # #transform.append(transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]))  # CiFar10

        transform = transforms.Compose(
            [
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomResizedCrop(size=imgsize),
                transforms.Resize(size=imgsize),
                transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                # transforms.GaussianBlur(kernel_size=3),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        self.transform = transform #transforms.Compose(transform)

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.num_views)]


class SimCLREvalTransform():
    def __init__(self, imgsize, num_views=2):
        self.num_views = num_views
        self.transform = transforms.Compose([
            transforms.Resize(imgsize),
            transforms.ToTensor(),
            #transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])  # CiFar10
            transforms.Normalize(0.5, 0.5)  # CiFar10
        ])

    def __call__(self, x):
        return [self.transform(x) for _ in range(self.num_views)] if self.num_views > 1 else self.transform(x)