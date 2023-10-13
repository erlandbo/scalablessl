import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch


# TODO add plots
class SCLVisualizer():
    def __init__(self, device, sclmodel, plotname=""):
        super().__init__()
        self.plotname = plotname
        self.device = device
        self.sclmodel = sclmodel.to(self.device)

    @torch.no_grad()
    def fit(self, trainloader, testloader):
        self.sclmodel.eval()
        X_train, X_test, y_train, y_test = [], [], [], []
        for X, y in trainloader:
            train_embeds = self.sclmodel(X.to(self.device))
            X_train.append(train_embeds.detach().cpu().numpy())
            y_train.append(y.detach().numpy())
        for X, y in testloader:
            test_embeds = self.sclmodel(X.to(self.device))
            X_test.append(test_embeds.detach().cpu().numpy())
            y_test.append(y.detach().numpy())

        X_train, y_train = np.concatenate(X_train), np.concatenate(y_train)
        X_test, y_test = np.concatenate(X_test), np.concatenate(y_test)

        # print(X_train.shape)
        # print(X_test.shape)
        # print(y_train.shape)
        # print(y_test.shape)

        # train_df = pd.DataFrame(data={
        #     "train_embeds": X_train.tolist(),
        #     "train_labels": y_train.tolist(),
        # })
        #
        # test_df = pd.DataFrame(data={
        #     "test_embeds": X_test.tolist(),
        #     "test_labels": y_test.tolist()
        # })
        #
        # train_df.to_csv("embeds/scl_train_embeds_{}.csv".format(self.plotname))
        # test_df.to_csv("embeds/scl_test_embeds_{}.csv".format(self.plotname))

        np.savez("embeds/scl_train_embeds_{}.csv".format(self.plotname), data=X_train, labels=y_train)
        np.savez("embeds/scl_test_embeds_{}.csv".format(self.plotname), data=X_test, labels=y_test)


    @torch.no_grad()
    def tensorboard_projector(self, dataloader, mu, sigma, tag=""):
        writer = SummaryWriter(log_dir="embeds/" + self.plotname + "_" + tag)
        mu = torch.tensor(mu)[:, None, None]
        sigma = torch.tensor(sigma)[:, None, None]
        self.sclmodel.eval()
        X_images, X_embeds, X_labels = [], [], []
        for idx, (imgs, labels) in enumerate(dataloader):
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            embeds = self.sclmodel(imgs)

            X_images.append(imgs.detach().cpu()* sigma + mu)
            X_embeds.append(embeds.detach().cpu().numpy())
            X_labels.extend(labels.detach().cpu().numpy())

        X_images = torch.cat(X_images)
        X_embeds = np.concatenate(X_embeds)
        # X_labels = np.concatenate(X_labels)

        writer.add_embedding(
            X_embeds,
            metadata=X_labels,
            label_img=X_images,
            tag="embeddings"
        )
