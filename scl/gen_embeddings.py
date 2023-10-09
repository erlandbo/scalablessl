from torch.utils.tensorboard import SummaryWriter
import torch
from scl import SCL
from utils import load_imagedataset
from torch.utils.data import DataLoader
from Datasets import SCLFinetuneDataset
from ImageAugmentations import SCLEvalTransform


def compute_embeddings(model, dataloader, logpath="embeds"):
    writer = SummaryWriter(log_dir=logpath)
    mu, sig = torch.tensor([0.4914, 0.4822, 0.4465]), torch.tensor([0.2023, 0.1994, 0.2010])
    model.eval()
    for idx, (imgs, labels) in enumerate(dataloader):
        imgs, labels = imgs.to(model.device), labels.to(model.device)
        with torch.no_grad():
            embeds = model(imgs)
        writer.add_embedding(
            embeds.detach().cpu().numpy(),
            # metadata=labels.detach().cpu(),
            label_img=imgs.detach().cpu() * sig[None, :, None, None] + mu[None , :, None, None],
            global_step=idx,
            tag="embedding"
        )


if __name__ == "__main__":
    DATASET = "cifar10"
    IMGSIZE = 32
    BATCHSIZE = 512
    NUM_WORKERS = 20
    LOG_PATH = "tb_logs/scl/embeds"

    checkpoint_path = "tb_logs/"


    model = SCL.load_from_checkpoint(checkpoint_path)

    traindataset, valdataset, testdataset, mean, std, num_classes = load_imagedataset(datasetname=DATASET, val_split=0.05)

    trainloader = DataLoader(
        SCLFinetuneDataset(
            traindataset,
            transform=SCLEvalTransform(imgsize=IMGSIZE, mean=mean, std=std, num_views=1)
        ),
        batch_size=BATCHSIZE,
        num_workers=NUM_WORKERS,
    )
    valloader = DataLoader(
        SCLFinetuneDataset(
            valdataset,
            transform=SCLEvalTransform(imgsize=NUM_WORKERS, mean=mean, std=std, num_views=1)
        ),
        batch_size=BATCHSIZE,
        num_workers=NUM_WORKERS,
    )

    testloader = DataLoader(
        SCLFinetuneDataset(
            testdataset,
            transform=SCLEvalTransform(imgsize=IMGSIZE, mean=mean, std=std, num_views=1)
        ),
        batch_size=BATCHSIZE,
        num_workers=NUM_WORKERS,
    )

    compute_embeddings(model, dataloader=testloader, logpath=LOG_PATH)
