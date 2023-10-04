from sklearn.neighbors import KNeighborsClassifier
import math
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch


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


