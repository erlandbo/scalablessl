import numpy as np
from scl import SCL
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import load_knndataset
from scipy.spatial.distance import cdist


# TODO Avoid out of memory
@torch.no_grad()
def build_simmatrix(model, dataloader, plotname):
    embeds, labels = [], []
    model.eval()
    for X, y in dataloader:
        embeds.append(model(X.to(model.device)).detach().cpu().numpy())
        labels.append(y.detach().numpy())
    embeds, labels = np.concatenate(embeds), np.concatenate(labels)
    labels_order = np.argsort(labels)
    embeds = embeds[labels_order]
    N = embeds.shape[0]
    simmat = embeds @ embeds.T
    plt.spy(simmat)
    plt.imsave(f"plots/{plotname}.png", format="png")


if __name__ == "__main__":
    CHECKPOINT_PATH = "tb_logs/scl/version_0/checkpoints/epoch=288-step=56644.ckpt"
    DATASET = "cifar10"
    IMGSIZE = 32
    NUM_WORKERS = 20

    PLOTNAME = "cifar10_batch256"

    plaintraindataset, plainvaldataset = load_knndataset(name=DATASET, imgsize=IMGSIZE)
    trainloader = DataLoader(dataset=plaintraindataset, batch_size=512, shuffle=True, num_workers=NUM_WORKERS)
    valloader = DataLoader(dataset=plainvaldataset, batch_size=512, shuffle=False, num_workers=NUM_WORKERS)

    model = SCL.load_from_checkpoint(checkpoint_path=CHECKPOINT_PATH)

    # build_simmatrix(model, trainloader, plotname=PLOTNAME + "_train")
    build_simmatrix(model, trainloader, PLOTNAME + "_val")

